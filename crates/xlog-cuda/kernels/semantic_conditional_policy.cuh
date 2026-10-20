#pragma once

// Included after the native categorical types and exact U192 arithmetic.
// Legality, stable score ordering and canonical category order have one owner
// in the enclosing kernel. This is their common conditional probability law.
namespace semantic_conditional_policy {

__device__ bool prepare_conditional_cell(
        const float* row,const uint32_t* sorted,uint32_t count,uint8_t* active_sets,
        Receipt& result,float& maximum,U192& g,U192& active_sum,
        semantic_graph::NativeWorkTally* work) {
    if(!count)return false;
    result.legal_count=count;
    maximum=row[sorted[0]];
    g=U192{};
    active_sum=U192{};
    if(count==1) {
        result.choice=sorted[0];
        result.p.w[0]=result.q.w[0]=result.factor_denominator.w[0]=1;
        result.active_count=1;
        result.cdf_end=result.mass=uint64_t(1)<<63;
        if(active_sets)active_sets[sorted[0]]=1;
        return true;
    }

    g=sub(shifted(1,149),shifted(count,118));
    U192 prefix{};
    uint32_t h=0;
    for(uint32_t i=0;i<count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CdfStep,1);
        const U192 d=distance_scaled(maximum,row[sorted[i]]);
        prefix=add(prefix,d);
        if(compare(multiply(d,i+1),add(prefix,g))<0) {
            h=i+1;
            active_sum=prefix;
        }
    }
    result.active_count=h;
    result.q=shifted(h,149);
    if(active_sets)for(uint32_t i=0;i<h;++i)active_sets[sorted[i]]=1;
    return true;
}

// The caller restores original category order between projection and CDF,
// preserving the sampler's numerical and native-work event order.
__device__ bool retain_selected_probability(
        const float* row,const uint32_t* legal,uint32_t count,
        bool fixed_choice,uint32_t demonstrated_choice,Receipt& result,
        float maximum,U192 g,U192 active_sum,semantic_graph::NativeWorkTally* work) {
    if(!count)return false;
    if(count==1)return !fixed_choice || demonstrated_choice==result.choice;
    const uint32_t h=result.active_count;
    U192 cumulative{};
    uint64_t lower=0;
    for(uint32_t index=0;index<count;++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CdfStep,1);
        const uint32_t category=legal[index];
        const U192 base=add(active_sum,g);
        const U192 hd=multiply(distance_scaled(maximum,row[category]),h);
        U192 p=shifted(h,118);
        if(compare(base,hd)>0)p=add(p,sub(base,hd));
        const U192 below=cumulative;
        cumulative=add(cumulative,p);
        const uint64_t upper=boundary(cumulative,h);
        const bool selected=fixed_choice ? category==demonstrated_choice
            : result.draw>=lower && result.draw<upper;
        if(selected) {
            result.choice=category;
            result.p=p;
            result.cdf_start=lower;
            result.cdf_end=upper;
            result.selected_rank=index;
            result.cumulative_mass_below=below;
            result.mass=upper-lower;
            result.factor_denominator=multiply(shifted(h,86),result.mass);
        }
        lower=upper;
    }
    return compare(cumulative,result.q)==0 && lower==(uint64_t(1)<<63) && result.mass!=0;
}

} // namespace semantic_conditional_policy
