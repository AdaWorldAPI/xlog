#ifndef XLOG_SEMANTIC_MODEL_FORWARD_STREAM_CUH
#define XLOG_SEMANTIC_MODEL_FORWARD_STREAM_CUH

// Included by semantic_transition.cu after the publication, training-view,
// scalar-logit and native-work definitions. Blocks are consumed before the
// producer reuses its contiguous scratch allocation. No logits survive here.
struct ModelForwardStreamState {
    uint64_t abi,phase,row,position,row_count,capacity,vocabulary,scalar_type;
    uint64_t block_positions,role,generation,publication_word,finite,blocks;
    uint64_t row_digest[4],logits_digest[4],selection_identity[4];
    semantic_graph::NativeWorkTally work;
};
struct ModelForwardStreamInputs {
    uint64_t control,lease,selection,state,row_seals,receipt;
    uint64_t row_count,capacity,vocabulary,scalar_type,block_positions,role;
    uint64_t row,position_begin,count,logits,position_stride_bytes;
    uint64_t candidate_seals,candidate_seal_count,work_sequence;
    uint64_t leaf_scratch,leaf_capacity;
};
struct ModelStreamLeaf {
    uint64_t digest[4],finite;
    semantic_graph::NativeWorkTally work;
};
struct ModelCanaryStreamRow {
    uint64_t row_ordinal,kind,reason,obligations,baseline_correct,candidate_correct,lost;
    uint64_t retention_denominator,protected_seen,protected_lost,positions,maximum_bits;
    uint64_t baseline_digest[4],candidate_digest[4],identity[4];
};
struct ModelCanaryStreamState {
    uint64_t abi,phase,canary_index,position,row_count,capacity,vocabulary,scalar_type;
    uint64_t block_positions,publication_word,selection_identity[4];
    ModelCanaryStreamRow rows[5];
    semantic_graph::NativeWorkTally work;
};
struct ModelCanaryStreamPosition {
    ModelCanaryStreamRow metrics;
    semantic_graph::NativeWorkTally work;
};
struct ModelCanaryStreamReceipt {
    uint64_t abi,availability,row_count,capacity,vocabulary;
    uint64_t baseline_identity[4],candidate_identity[4],selection_identity[4];
    uint64_t summary_digest[4],identity[4];
};
struct ModelCanaryStreamInputs {
    uint64_t control,lease,selection,roster_rows,objective,canaries,protected_members;
    uint64_t protected_member_count,task,task_words,retention_labels,kinds;
    uint64_t state,receipt,baseline_state,candidate_state;
    uint64_t baseline_row_seals,candidate_row_seals,forward_receipts;
    uint64_t canary_index,position_begin,count,baseline_logits,candidate_logits;
    uint64_t position_stride_bytes;
    uint64_t baseline_leaf_scratch,candidate_leaf_scratch,leaf_capacity,position_scratch;
};
static_assert(sizeof(ModelForwardStreamState)==296,"model forward stream state ABI");
static_assert(sizeof(ModelForwardStreamInputs)==176,"model forward stream inputs ABI");
static_assert(sizeof(ModelStreamLeaf)==128,"model stream leaf ABI");
static_assert(sizeof(ModelCanaryStreamRow)==192,"model canary stream row ABI");
static_assert(sizeof(ModelCanaryStreamState)==1160,"model canary stream state ABI");
static_assert(sizeof(ModelCanaryStreamPosition)==280,"model canary stream position ABI");
static_assert(sizeof(ModelCanaryStreamReceipt)==200,"model canary stream receipt ABI");
static_assert(sizeof(ModelCanaryStreamInputs)==232,"model canary stream inputs ABI");

constexpr uint64_t MODEL_STREAM_LEAF_BYTES=4096;

__device__ void semantic_stream_seed(uint64_t row,uint64_t role,uint64_t rows,
        uint64_t capacity,uint64_t vocabulary,uint64_t scalar_type,uint64_t* digest,
        semantic_graph::NativeWorkTally* work) {
    const uint64_t words[]={0x6d61657274736c6dULL,1,row,role,rows,capacity,vocabulary,scalar_type};
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(words),sizeof(words),digest,work);
}

struct ModelStreamLeafBytes {
    uint64_t prefix[8];
    const uint8_t* values;
    semantic_graph::NativeWorkTally* work;
    __device__ uint8_t operator[](uint64_t index) const {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,1);
        return index<sizeof(prefix) ? reinterpret_cast<const uint8_t*>(prefix)[index] :
            values[index-sizeof(prefix)];
    }
};

struct ModelStreamDigestBytes {
    uint64_t prefix[8];
    const ModelStreamLeaf* leaves;
    semantic_graph::NativeWorkTally* work;
    __device__ uint8_t operator[](uint64_t index) const {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,1);
        if(index<sizeof(prefix))return reinterpret_cast<const uint8_t*>(prefix)[index];
        index-=sizeof(prefix);
        return reinterpret_cast<const uint8_t*>(leaves[index/32].digest)[index%32];
    }
};

__device__ void semantic_stream_merge_native(semantic_graph::NativeWorkTally* destination,
        const semantic_graph::NativeWorkTally& source) {
    uint64_t sum=0;
    for(uint32_t event=0;event<9;++event) {
        semantic_graph::charge_native(destination,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(source.events[event]>UINT64_MAX-sum) { destination->overflow=1;return; }
        sum+=source.events[event];
    }
    if(source.overflow || sum!=source.units) { destination->overflow=1;return; }
    for(uint32_t event=0;event<9;++event) {
        semantic_graph::charge_native(destination,semantic_graph::NativeWorkEvent::TableSlot,1);
        semantic_graph::charge_native(destination,static_cast<semantic_graph::NativeWorkEvent>(event),source.events[event]);
    }
}

__device__ void semantic_stream_leaf(uint64_t pointer,uint64_t bytes,uint64_t row,
        uint64_t begin,uint64_t count,uint64_t role,uint64_t scalar_type,uint64_t index,
        ModelStreamLeaf* destination,bool command) {
    ModelStreamLeaf leaf{};leaf.finite=1;
    auto* work=&leaf.work;
    if(index==0 && command)semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    const uint64_t offset=index*MODEL_STREAM_LEAF_BYTES;
    const uint64_t length=bytes-offset<MODEL_STREAM_LEAF_BYTES ? bytes-offset : MODEL_STREAM_LEAF_BYTES;
    const uint64_t element=scalar_type==6 ? 4 : 2;
    for(uint64_t byte=0;byte<length;byte+=element) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,element);
        // IEEE finite classification needs no conversion, exponentiation or
        // reduction. It is identical to isfinite on the canonical scalar load.
        const auto* source=reinterpret_cast<const uint8_t*>(pointer+offset+byte);
        if(scalar_type==6) {
            if((*reinterpret_cast<const uint32_t*>(source)&0x7f800000U)==0x7f800000U)leaf.finite=0;
        } else {
            const uint16_t bits=*reinterpret_cast<const uint16_t*>(source);
            const uint16_t exponent=scalar_type==5 ? 0x7f80U : 0x7c00U;
            if((bits&exponent)==exponent)leaf.finite=0;
        }
    }
    ModelStreamLeafBytes input{};
    input.prefix[0]=0x6661656c6d616572ULL;input.prefix[1]=1;input.prefix[2]=row;
    input.prefix[3]=begin;input.prefix[4]=count;input.prefix[5]=role;
    input.prefix[6]=index;input.prefix[7]=length;
    input.values=reinterpret_cast<const uint8_t*>(pointer+offset);input.work=work;
    semantic_graph::sha256(input,sizeof(input.prefix)+length,leaf.digest,work);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(leaf));
    destination[index]=leaf;
}

__device__ bool semantic_stream_block_geometry(uint64_t capacity,uint64_t vocabulary,
        uint64_t scalar_type,uint64_t block_positions,uint64_t begin,uint64_t count,
        uint64_t pointer,uint64_t stride,uint64_t* bytes) {
    const uint64_t element=scalar_type==6 ? 4 : 2;
    if((scalar_type!=4 && scalar_type!=5 && scalar_type!=6) || !capacity ||
       !vocabulary || !block_positions || begin>=capacity ||
       begin%block_positions || count!=(capacity-begin<block_positions ?
           capacity-begin : block_positions) || vocabulary>UINT64_MAX/element ||
       stride!=vocabulary*element || count>UINT64_MAX/stride || !pointer ||
       pointer%element || pointer>UINT64_MAX-count*stride ||
       count*stride>UINT64_MAX/8-64)return false;
    *bytes=count*stride;return true;
}

__device__ void semantic_stream_fold_block(const ModelStreamLeaf* leaves,uint64_t leaf_count,
        uint64_t bytes,uint64_t row,uint64_t begin,uint64_t count,uint64_t* digest,
        semantic_graph::NativeWorkTally* work) {
    ModelStreamDigestBytes input{};
    for(uint32_t word=0;word<4;++word)input.prefix[word]=digest[word];
    input.prefix[4]=row;input.prefix[5]=begin;input.prefix[6]=count;input.prefix[7]=bytes;
    input.leaves=leaves;input.work=work;
    semantic_graph::sha256(input,sizeof(input.prefix)+leaf_count*32,digest,work);
}

__device__ void semantic_stream_copy_identity(uint64_t* destination,const uint64_t* source,
        semantic_graph::NativeWorkTally* work) {
    semantic_graph::copy_identity(destination,source,work);
}

extern "C" __global__ void semantic_model_forward_stream_begin(ModelForwardStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    if(!inputs.control || !inputs.lease || !inputs.selection || !inputs.state ||
       !inputs.row_seals || !inputs.receipt || inputs.role>2 || !inputs.row_count ||
       !inputs.capacity || !inputs.vocabulary || !inputs.block_positions ||
       (inputs.scalar_type!=4 && inputs.scalar_type!=5 && inputs.scalar_type!=6)) {
        semantic_content_integrity_trap();return;
    }
    auto& state=*reinterpret_cast<ModelForwardStreamState*>(inputs.state);
    // This device reset, not the host capture cursor, owns EVERY graph replay.
    state=ModelForwardStreamState{};
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::Command,1);
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(state));
    const auto& control=*reinterpret_cast<const PublicationControl*>(inputs.control);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(inputs.lease);
    const auto* base=publication_acquired_bank(control,lease,&state.work);
    const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(inputs.selection);
    if(!base || publication_validate_selected_model(control,*base,&state.work) ||
       (inputs.role!=2 && base->header.model_generation>=UINT32_MAX) || selection.status ||
       !selection.row_count || selection.row_count>inputs.row_count ||
       selection.capacity!=inputs.capacity) { semantic_content_integrity_trap();return; }
    state.abi=1;state.phase=1;state.finite=1;state.row_count=inputs.row_count;
    state.capacity=inputs.capacity;state.vocabulary=inputs.vocabulary;
    state.scalar_type=inputs.scalar_type;state.block_positions=inputs.block_positions;
    state.role=inputs.role;state.generation=base->header.model_generation+(inputs.role==1 ? 1 : 0);
    state.publication_word=base->header.publication_word;
    semantic_stream_copy_identity(state.selection_identity,selection.identity,&state.work);
    semantic_stream_seed(0,inputs.role,inputs.row_count,inputs.capacity,
        inputs.vocabulary,inputs.scalar_type,state.row_digest,&state.work);
    semantic_stream_seed(UINT64_MAX,inputs.role,inputs.row_count,inputs.capacity,
        inputs.vocabulary,inputs.scalar_type,state.logits_digest,&state.work);
    auto* rows=reinterpret_cast<uint64_t*>(inputs.row_seals);
    for(uint64_t word=0;word<inputs.row_count*4;++word)
        rows[word]=semantic_graph::written_word(0,&state.work);
    *reinterpret_cast<ModelForwardReceipt*>(inputs.receipt)=ModelForwardReceipt{};
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(ModelForwardReceipt));
}

extern "C" __global__ void semantic_model_forward_stream_block_leaves(ModelForwardStreamInputs inputs) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto& state=*reinterpret_cast<const ModelForwardStreamState*>(inputs.state);
    uint64_t bytes=0;
    if(state.abi!=1 || state.phase!=1 || inputs.row!=state.row ||
       inputs.position_begin!=state.position || state.row>=state.row_count ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.logits,
           inputs.position_stride_bytes,&bytes)) { semantic_content_integrity_trap();return; }
    const uint64_t leaves=(bytes+MODEL_STREAM_LEAF_BYTES-1)/MODEL_STREAM_LEAF_BYTES;
    if(!inputs.leaf_scratch || leaves>inputs.leaf_capacity) { semantic_content_integrity_trap();return; }
    if(index>=leaves)return;
    semantic_stream_leaf(inputs.logits,bytes,state.row,inputs.position_begin,inputs.count,
        state.role,state.scalar_type,index,reinterpret_cast<ModelStreamLeaf*>(inputs.leaf_scratch),true);
}

extern "C" __global__ void semantic_model_forward_stream_block(ModelForwardStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    auto& state=*reinterpret_cast<ModelForwardStreamState*>(inputs.state);
    auto* work=&state.work;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    uint64_t bytes=0;
    if(state.abi!=1 || state.phase!=1 || inputs.row!=state.row ||
       inputs.position_begin!=state.position || state.row>=state.row_count ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.logits,
           inputs.position_stride_bytes,&bytes)) { semantic_content_integrity_trap();return; }
    const uint64_t leaf_count=(bytes+MODEL_STREAM_LEAF_BYTES-1)/MODEL_STREAM_LEAF_BYTES;
    if(!inputs.leaf_scratch || leaf_count>inputs.leaf_capacity) { semantic_content_integrity_trap();return; }
    const auto* leaves=reinterpret_cast<const ModelStreamLeaf*>(inputs.leaf_scratch);
    // Integer finite flags and work counts combine exactly. All scalar values,
    // including PAD and unselected rows, were inspected by the parallel leaves.
    for(uint64_t leaf=0;leaf<leaf_count;++leaf) {
        semantic_stream_merge_native(work,leaves[leaf].work);
        if(!leaves[leaf].finite)state.finite=0;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    }
    semantic_stream_fold_block(leaves,leaf_count,bytes,state.row,inputs.position_begin,
        inputs.count,state.row_digest,work);
    state.position+=inputs.count;++state.blocks;
    if(state.position==state.capacity) {
        auto* rows=reinterpret_cast<uint64_t*>(inputs.row_seals);
        semantic_stream_copy_identity(rows+state.row*4,state.row_digest,work);
        publication_fold(state.logits_digest,state.row,state.capacity,state.row_digest,work);
        state.position=0;++state.row;
        if(state.row<state.row_count)
            semantic_stream_seed(state.row,state.role,state.row_count,state.capacity,
                state.vocabulary,state.scalar_type,state.row_digest,work);
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,4*sizeof(uint64_t));
}

// The only candidate numerical digest is the original, already-recorded output
// content witness. Its cell addresses are immutable cold metadata; their values
// are produced BEFORE this finish node in the captured graph.
__device__ bool semantic_stream_model_numerical(const PublicationControl& control,
        const PublicationBank& base,const ModelForwardStreamInputs& inputs,
        uint64_t* numerical,semantic_graph::NativeWorkTally* work) {
    if(inputs.role!=1) {
        semantic_stream_copy_identity(numerical,base.header.model_numerical_digest,work);return true;
    }
    if(!inputs.candidate_seals || !inputs.candidate_seal_count)return false;
    const uint64_t directory=control.directories[base.header.publication_word&1];
    if(!directory || directory%alignof(PublicationRange))return false;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(directory);
    const auto* seals=reinterpret_cast<const ModelForwardSealInput*>(inputs.candidate_seals);
    const char domain[]="xlog.semantic.model-numerics.v1";
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(domain),sizeof(domain),numerical,work);
    publication_fold(numerical,0,0,base.header.model_geometry_digest,work);
    uint64_t seal_index=0;
    for(uint64_t item=0;item<base.header.range_count;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& range=ranges[item];
        if(range.role<18 || range.role>25)continue;
        if(seal_index>=inputs.candidate_seal_count)return false;
        const auto& seal=seals[seal_index++];
        if(seal.role!=range.role || seal.index!=range.index || !seal.tensor_digest ||
           seal.tensor_digest%alignof(uint64_t) || !seal.backing_digest ||
           seal.backing_digest%alignof(uint64_t))return false;
        publication_fold(numerical,seal.role,seal.index,
            reinterpret_cast<const uint64_t*>(seal.tensor_digest),work);
        publication_fold(numerical,seal.role,seal.index,
            reinterpret_cast<const uint64_t*>(seal.backing_digest),work);
    }
    return seal_index==inputs.candidate_seal_count;
}

extern "C" __global__ void semantic_model_forward_stream_finish(ModelForwardStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    auto& state=*reinterpret_cast<ModelForwardStreamState*>(inputs.state);
    auto* work=&state.work;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    const auto& control=*reinterpret_cast<const PublicationControl*>(inputs.control);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(inputs.lease);
    const auto* base=publication_acquired_bank(control,lease,work);
    if(!base || state.abi!=1 || state.phase!=1 || state.row!=state.row_count ||
       state.position || state.publication_word!=base->header.publication_word ||
       state.generation!=base->header.model_generation+(state.role==1 ? 1 : 0) || !inputs.work_sequence ||
       inputs.role!=state.role) { semantic_content_integrity_trap();return; }
    ModelForwardReceipt receipt{};
    receipt.abi=2;receipt.availability=2;receipt.role=state.role;
    receipt.generation=state.generation;receipt.work_sequence=inputs.work_sequence;
    // Stream identity replaces a fictitious dense pointer. The owner is retained
    // until the original graph's known terminal retirement, including failures.
    receipt.stream_state=inputs.state;receipt.scalar_type=state.scalar_type;
    receipt.row_count=state.row_count;receipt.capacity=state.capacity;receipt.vocabulary=state.vocabulary;
    receipt.block_positions=state.block_positions;
    receipt.row_seals=inputs.row_seals;receipt.finite=state.finite;
    semantic_stream_copy_identity(receipt.model_geometry_digest,base->header.model_geometry_digest,work);
    if(!semantic_stream_model_numerical(control,*base,inputs,receipt.model_numerical_digest,work)) {
        semantic_content_integrity_trap();return;
    }
    semantic_stream_copy_identity(receipt.logits_digest,state.logits_digest,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipt),
        sizeof(receipt)-sizeof(receipt.identity),receipt.identity,work);
    *reinterpret_cast<ModelForwardReceipt*>(inputs.receipt)=receipt;state.phase=2;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(receipt)+sizeof(state.phase));
}

__device__ bool semantic_stream_forward_valid(const ModelForwardStreamState& state,
        const ModelForwardReceipt& receipt,uint64_t pointer,uint64_t row_seals,
        uint64_t role,const PublicationHeader& baseline,semantic_graph::NativeWorkTally* work) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(state)+sizeof(receipt));
    if(state.abi!=1 || state.phase!=2 || state.row!=state.row_count || state.position ||
       state.role!=role || state.publication_word!=baseline.publication_word ||
       state.generation!=baseline.model_generation+(role==1 ? 1 : 0) || receipt.abi!=2 ||
       receipt.availability!=2 || receipt.role!=role || receipt.stream_state!=pointer ||
       receipt.generation!=state.generation || !receipt.work_sequence ||
       receipt.scalar_type!=state.scalar_type || receipt.row_count!=state.row_count ||
       receipt.capacity!=state.capacity || receipt.vocabulary!=state.vocabulary ||
       receipt.block_positions!=state.block_positions || receipt.row_seals!=row_seals ||
       receipt.finite!=state.finite ||
       !publication_identity_equal(receipt.logits_digest,state.logits_digest,work) ||
       !publication_identity_equal(receipt.model_geometry_digest,baseline.model_geometry_digest,work) ||
       (role!=1 && !publication_identity_equal(receipt.model_numerical_digest,
           baseline.model_numerical_digest,work)))return false;
    uint64_t identity[4];
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipt),
        sizeof(receipt)-sizeof(receipt.identity),identity,work);
    return publication_identity_equal(identity,receipt.identity,work);
}

// Evaluation's original wrapper hashes losses/admissibility/slab AFTER its
// enqueue callback. Bind those existing device seals here, never a duplicate
// output hash or an unavailable future candidate-style numerical seal.
struct ModelEvaluationStreamBinding {
    uint64_t abi,forward_identity[4],output_digests[3][4],identity[4];
};
struct ModelEvaluationStreamInputs {
    uint64_t control,lease,selection,state,row_seals,receipt,binding;
    uint64_t output_digests[3],admissible,gate,native_work;
};
static_assert(sizeof(ModelEvaluationStreamBinding)==168,"evaluation stream binding ABI");
static_assert(sizeof(ModelEvaluationStreamInputs)==104,"evaluation stream input ABI");

extern "C" __global__ void semantic_model_evaluation_stream_binding(
        ModelEvaluationStreamInputs inputs,uint64_t verify) {
    if(blockIdx.x || threadIdx.x)return;
    if(!inputs.native_work) { semantic_content_integrity_trap();return; }
    auto* target=reinterpret_cast<semantic_graph::NativeWorkTally*>(inputs.native_work);
    semantic_graph::NativeWorkTally work{};
    semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::Command,1);
    const auto& state=*reinterpret_cast<const ModelForwardStreamState*>(inputs.state);
    const auto& receipt=*reinterpret_cast<const ModelForwardReceipt*>(inputs.receipt);
    const bool valid=[&]() {
        const auto& control=*reinterpret_cast<const PublicationControl*>(inputs.control);
        const auto& lease=*reinterpret_cast<const PublicationLease*>(inputs.lease);
        const auto* base=publication_acquired_bank(control,lease,&work);
        const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(inputs.selection);
        if(!base || verify>1 || selection.status || state.finite>1 ||
           !semantic_stream_forward_valid(state,receipt,inputs.state,inputs.row_seals,
               2,base->header,&work) ||
           !publication_identity_equal(state.selection_identity,selection.identity,&work))return false;
        for(uint32_t output=0;output<3;++output)
            if(!inputs.output_digests[output] || inputs.output_digests[output]%alignof(uint64_t))return false;
        auto& stored=*reinterpret_cast<ModelEvaluationStreamBinding*>(inputs.binding);
        if(!verify) {
            ModelEvaluationStreamBinding binding{};binding.abi=1;
            semantic_stream_copy_identity(binding.forward_identity,receipt.identity,&work);
            for(uint32_t output=0;output<3;++output)
                semantic_stream_copy_identity(binding.output_digests[output],
                    reinterpret_cast<const uint64_t*>(inputs.output_digests[output]),&work);
            semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&binding),
                sizeof(binding)-sizeof(binding.identity),binding.identity,&work);
            stored=binding;
            semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(binding));
        } else {
            semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(stored));
            if(stored.abi!=1 || !inputs.admissible || !inputs.gate ||
               !publication_identity_equal(stored.forward_identity,receipt.identity,&work))return false;
            for(uint32_t output=0;output<3;++output)
                if(!publication_identity_equal(stored.output_digests[output],
                    reinterpret_cast<const uint64_t*>(inputs.output_digests[output]),&work))return false;
            uint64_t identity[4];
            semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&stored),
                sizeof(stored)-sizeof(stored.identity),identity,&work);
            if(!publication_identity_equal(identity,stored.identity,&work))return false;
            const uint8_t admissible=*reinterpret_cast<const uint8_t*>(inputs.admissible);
            semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::CanonicalByte,1);
            if(admissible>1)return false;
            *reinterpret_cast<uint64_t*>(inputs.gate)=semantic_graph::written_word(
                admissible && state.finite,&work);
        }
        return true;
    }();
    // The full producer's independent tally enters its original cold attempt
    // exactly once per actual replay, never again at result verification.
    if(!verify)semantic_stream_merge_native(&work,state.work);
    semantic_stream_merge_native(target,work);
    if(!valid || work.overflow || target->overflow) {
        target->overflow=1;semantic_content_integrity_trap();
    }
}

extern "C" __global__ void semantic_model_canary_stream_begin(ModelCanaryStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    auto& state=*reinterpret_cast<ModelCanaryStreamState*>(inputs.state);
    state=ModelCanaryStreamState{};
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::Command,1);
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(state));
    const auto& control=*reinterpret_cast<const PublicationControl*>(inputs.control);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(inputs.lease);
    const auto* base=publication_acquired_bank(control,lease,&state.work);
    const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(inputs.selection);
    const auto& baseline=*reinterpret_cast<const ModelForwardStreamState*>(inputs.baseline_state);
    const auto& candidate=*reinterpret_cast<const ModelForwardStreamState*>(inputs.candidate_state);
    const auto* receipts=reinterpret_cast<const ModelForwardReceipt*>(inputs.forward_receipts);
    if(!base || selection.status ||
       !semantic_stream_forward_valid(baseline,receipts[0],inputs.baseline_state,
           inputs.baseline_row_seals,0,base->header,&state.work) ||
       !semantic_stream_forward_valid(candidate,receipts[1],inputs.candidate_state,
           inputs.candidate_row_seals,1,base->header,&state.work) ||
       receipts[0].work_sequence>=receipts[1].work_sequence ||
       baseline.row_count!=candidate.row_count || baseline.capacity!=candidate.capacity ||
       baseline.vocabulary!=candidate.vocabulary || baseline.scalar_type!=candidate.scalar_type ||
       baseline.block_positions!=candidate.block_positions ||
       !publication_identity_equal(baseline.selection_identity,selection.identity,&state.work) ||
       !publication_identity_equal(candidate.selection_identity,selection.identity,&state.work)) {
        semantic_content_integrity_trap();return;
    }
    state.abi=1;state.phase=1;state.row_count=baseline.row_count;
    state.capacity=baseline.capacity;state.vocabulary=baseline.vocabulary;
    state.scalar_type=baseline.scalar_type;state.block_positions=baseline.block_positions;
    state.publication_word=baseline.publication_word;
    semantic_stream_copy_identity(state.selection_identity,selection.identity,&state.work);
    const auto* canaries=reinterpret_cast<const SemanticTrainingCanaryRecord*>(inputs.canaries);
    const auto* rows=reinterpret_cast<const SemanticTrainingRosterRow*>(inputs.roster_rows);
    for(uint64_t item=0;item<5;++item) {
        const auto& canary=canaries[item];
        if(canary.evaluator_abi!=3 || canary.kind!=item+1 ||
           canary.row_ordinal>=selection.row_count || canary.row_ordinal>=state.row_count) {
            semantic_content_integrity_trap();return;
        }
        const auto& row=rows[canary.row_ordinal];
        if(row.ordinal!=canary.row_ordinal || row.window>state.capacity ||
           !publication_identity_equal(row.identity,canary.row_identity,&state.work) ||
           !publication_identity_equal(row.content_identity,canary.row_content_identity,&state.work)) {
            semantic_content_integrity_trap();return;
        }
        if(canary.kind!=1 && canary.kind!=4 && canary.obligation_member_count) {
            semantic_content_integrity_trap();return;
        }
        if(canary.kind==2) {
            if(canary.protected_member_offset>inputs.protected_member_count ||
               canary.protected_member_count>inputs.protected_member_count-canary.protected_member_offset) {
                semantic_content_integrity_trap();return;
            }
            const auto* members=reinterpret_cast<const uint64_t*>(inputs.protected_members);
            const auto* labels=reinterpret_cast<const int64_t*>(inputs.retention_labels);
            uint64_t previous=UINT64_MAX;
            for(uint64_t member=0;member<canary.protected_member_count;++member) {
                semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const uint64_t position=members[canary.protected_member_offset+member];
                if(position>=row.window || (member && position<=previous) ||
                   labels[canary.row_ordinal*state.capacity+position]<0) {
                    semantic_content_integrity_trap();return;
                }
                previous=position;
            }
        } else if(canary.protected_member_count) { semantic_content_integrity_trap();return; }
        if(canary.kind==1 || canary.kind==4) {
            TaskProgramBank task_bank{};
            const auto* task=reinterpret_cast<const uint64_t*>(inputs.task);
            if(!task_program_bank(task,inputs.task_words,&task_bank)) {
                semantic_content_integrity_trap();return;
            }
            const uint64_t* goal=task_bank.task.goal_witness;
            for(uint32_t root=0;root<4;++root) {
                bool present=false;
                for(uint32_t word=0;word<4;++word) {
                    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    present|=goal[root*4+word]!=0;
                }
                if(!present) { semantic_content_integrity_trap();return; }
            }
            if(!goal[16] || !goal[17]) { semantic_content_integrity_trap();return; }
        }
        auto& output=state.rows[item];output.kind=canary.kind;output.row_ordinal=canary.row_ordinal;
        semantic_stream_copy_identity(output.identity,canary.identity,&state.work);
        semantic_stream_seed(canary.row_ordinal,0,state.row_count,state.capacity,
            state.vocabulary,state.scalar_type,output.baseline_digest,&state.work);
        semantic_stream_seed(canary.row_ordinal,1,state.row_count,state.capacity,
            state.vocabulary,state.scalar_type,output.candidate_digest,&state.work);
        semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::TableSlot,1);
    }
    *reinterpret_cast<ModelCanaryStreamReceipt*>(inputs.receipt)=ModelCanaryStreamReceipt{};
    semantic_graph::charge_native(&state.work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(ModelCanaryStreamReceipt));
}

__device__ bool semantic_stream_task_position(const ModelCanaryStreamInputs& inputs,
        const SemanticTrainingCanaryRecord& canary,const SemanticTrainingObjectiveRecord& objective,
        uint64_t position,uint64_t baseline_token,uint64_t candidate_token,
        ModelCanaryStreamRow& output,semantic_graph::NativeWorkTally* work) {
    if(objective.canary_source_offset>inputs.protected_member_count ||
       objective.canary_source_words>inputs.protected_member_count-objective.canary_source_offset ||
       objective.canary_source_words<13 || canary.obligation_member_offset>inputs.protected_member_count ||
       canary.obligation_member_count>inputs.protected_member_count-canary.obligation_member_offset)return false;
    const auto* members=reinterpret_cast<const uint64_t*>(inputs.protected_members);
    const auto* source=members+objective.canary_source_offset;
    const uint64_t queries=source[12];
    if(!queries || queries>(UINT64_MAX-13)/5 || objective.canary_source_words!=13+5*queries ||
       queries!=canary.obligation_member_count)return false;
    for(uint32_t identity=0;identity<3;++identity) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(!(source[identity*4]|source[identity*4+1]|source[identity*4+2]|source[identity*4+3]))return false;
    }
    for(uint64_t query=0;query<queries;++query) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(members[canary.obligation_member_offset+query]!=position)continue;
        const auto* statement=source+13+5*query;
        if(!(statement[0]|statement[1]|statement[2]|statement[3]) || statement[4]>3)return false;
        const uint64_t expected=objective.truth_tokens[statement[4]];
        const bool baseline_ok=baseline_token==expected,candidate_ok=candidate_token==expected;
        ++output.obligations;output.baseline_correct+=baseline_ok;
        output.candidate_correct+=candidate_ok;output.lost+=baseline_ok && !candidate_ok;
    }
    return true;
}

extern "C" __global__ void semantic_model_canary_stream_block_leaves(ModelCanaryStreamInputs inputs) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t role=blockIdx.y;
    const auto& state=*reinterpret_cast<const ModelCanaryStreamState*>(inputs.state);
    uint64_t bytes=0;
    const uint64_t pointer=role ? inputs.candidate_logits : inputs.baseline_logits;
    const uint64_t scratch=role ? inputs.candidate_leaf_scratch : inputs.baseline_leaf_scratch;
    if(role>1 || state.abi!=1 || state.phase!=1 || state.canary_index>=5 ||
       inputs.canary_index!=state.canary_index || inputs.position_begin!=state.position ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,pointer,
           inputs.position_stride_bytes,&bytes)) { semantic_content_integrity_trap();return; }
    const uint64_t leaves=(bytes+MODEL_STREAM_LEAF_BYTES-1)/MODEL_STREAM_LEAF_BYTES;
    if(!scratch || leaves>inputs.leaf_capacity) { semantic_content_integrity_trap();return; }
    if(index>=leaves)return;
    semantic_stream_leaf(pointer,bytes,state.rows[state.canary_index].row_ordinal,
        inputs.position_begin,inputs.count,role,state.scalar_type,index,
        reinterpret_cast<ModelStreamLeaf*>(scratch),role==0);
}

extern "C" __global__ void semantic_model_canary_stream_block_positions(ModelCanaryStreamInputs inputs) {
    const uint64_t local=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(local>=inputs.count)return;
    const auto& state=*reinterpret_cast<const ModelCanaryStreamState*>(inputs.state);
    ModelCanaryStreamPosition partial{};
    auto& output=partial.metrics;
    auto* work=&partial.work;
    if(local==0)semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    uint64_t bytes=0,candidate_bytes=0;
    if(state.abi!=1 || state.phase!=1 || state.canary_index>=5 ||
       inputs.canary_index!=state.canary_index || inputs.position_begin!=state.position ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.baseline_logits,
           inputs.position_stride_bytes,&bytes) ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.candidate_logits,
           inputs.position_stride_bytes,&candidate_bytes)) { semantic_content_integrity_trap();return; }
    output.row_ordinal=state.rows[state.canary_index].row_ordinal;
    output.kind=state.rows[state.canary_index].kind;
    output.reason=state.rows[state.canary_index].reason;
    const auto& canary=reinterpret_cast<const SemanticTrainingCanaryRecord*>(inputs.canaries)[state.canary_index];
    const auto& row=reinterpret_cast<const SemanticTrainingRosterRow*>(inputs.roster_rows)[output.row_ordinal];
    const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(inputs.objective);
    const auto* labels=reinterpret_cast<const int64_t*>(inputs.retention_labels);
    const auto* kinds=reinterpret_cast<const int64_t*>(inputs.kinds);
    const auto* members=reinterpret_cast<const uint64_t*>(inputs.protected_members);
    ModelUpdateCanaryInputs geometry{};geometry.scalar_type=state.scalar_type;
    geometry.vocabulary=state.vocabulary;geometry.position_stride_bytes=inputs.position_stride_bytes;
    geometry.vocabulary_stride_bytes=state.scalar_type==6 ? 4 : 2;
    ExecutionWork calculation{};
    // Each independent position keeps the canonical vocabulary iteration and
    // FP64 exp/sum order. Only integer counts and maxima combine afterward.
    for(uint32_t once=0;once<1;++once) {
        const uint64_t position=inputs.position_begin+local;
        if(position>=row.window || output.reason)continue;
        bool metric_position=false;
        int64_t retention_label=-1;
        if(canary.kind==1 || canary.kind==4) {
            if(canary.obligation_member_offset>inputs.protected_member_count ||
               canary.obligation_member_count>inputs.protected_member_count-canary.obligation_member_offset) {
                output.reason=7;continue;
            }
            for(uint64_t obligation=0;obligation<canary.obligation_member_count;++obligation) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                metric_position|=members[canary.obligation_member_offset+obligation]==position;
            }
        } else if(canary.kind==2) {
            retention_label=labels[output.row_ordinal*state.capacity+position];
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(retention_label));
            metric_position=retention_label>=0;
        } else if(canary.kind==3) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(int64_t));
            metric_position=kinds[output.row_ordinal*state.capacity+position]!=0;
        }
        if(!metric_position)continue;
        uint64_t baseline_token=0,candidate_token=0;
        if(canary.kind!=3 &&
           (!semantic_canary_argmax(geometry,inputs.baseline_logits,0,local,baseline_token,calculation,work) ||
            !semantic_canary_argmax(geometry,inputs.candidate_logits,0,local,candidate_token,calculation,work))) {
            output.reason=(canary.kind==1 || canary.kind==4) ? 7 : 1;continue;
        }
        if(canary.kind==1 || canary.kind==4) {
            if(!semantic_stream_task_position(inputs,canary,objective,position,
                    baseline_token,candidate_token,output,work))output.reason=7;
        } else if(canary.kind==2) {
            const int64_t label=retention_label;
            if(uint64_t(label)>=state.vocabulary) { output.reason=5;continue; }
            ++output.retention_denominator;
            const bool baseline_ok=baseline_token==uint64_t(label),candidate_ok=candidate_token==uint64_t(label);
            output.baseline_correct+=baseline_ok;output.candidate_correct+=candidate_ok;
            for(uint64_t member=0;member<canary.protected_member_count;++member) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(members[canary.protected_member_offset+member]==position) {
                    output.protected_lost+=baseline_ok && !candidate_ok;++output.protected_seen;
                }
            }
        } else if(canary.kind==3) {
            ++output.positions;
            double baseline_max=0.0,baseline_log=0.0,candidate_max=0.0,candidate_log=0.0;
            if(!semantic_canary_log_normalizer(geometry,inputs.baseline_logits,0,local,baseline_max,baseline_log,calculation,work) ||
               !semantic_canary_log_normalizer(geometry,inputs.candidate_logits,0,local,candidate_max,candidate_log,calculation,work)) {
                output.reason=1;continue;
            }
            double maximum=__longlong_as_double((long long)output.maximum_bits);
            for(uint64_t vocabulary=0;vocabulary<state.vocabulary;++vocabulary) {
                const uint64_t offset=local*inputs.position_stride_bytes+vocabulary*geometry.vocabulary_stride_bytes;
                const double baseline=__dsub_rn(__dsub_rn(semantic_canary_logit(inputs.baseline_logits,
                    state.scalar_type,offset),baseline_max),baseline_log);
                const double candidate=__dsub_rn(__dsub_rn(semantic_canary_logit(inputs.candidate_logits,
                    state.scalar_type,offset),candidate_max),candidate_log);
                const double difference=fabs(__dsub_rn(candidate,baseline));
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,2);
                if(!isfinite(difference)) { output.reason=1;break; }
                if(difference>maximum)maximum=difference;
            }
            output.maximum_bits=(uint64_t)__double_as_longlong(maximum);
        }
    }
    if(calculation.overflow)work->overflow=1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(partial));
    reinterpret_cast<ModelCanaryStreamPosition*>(inputs.position_scratch)[local]=partial;
}

extern "C" __global__ void semantic_model_canary_stream_block(ModelCanaryStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    auto& state=*reinterpret_cast<ModelCanaryStreamState*>(inputs.state);
    auto* work=&state.work;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    uint64_t bytes=0,candidate_bytes=0;
    if(state.abi!=1 || state.phase!=1 || state.canary_index>=5 ||
       inputs.canary_index!=state.canary_index || inputs.position_begin!=state.position ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.baseline_logits,
           inputs.position_stride_bytes,&bytes) ||
       !semantic_stream_block_geometry(state.capacity,state.vocabulary,state.scalar_type,
           state.block_positions,inputs.position_begin,inputs.count,inputs.candidate_logits,
           inputs.position_stride_bytes,&candidate_bytes)) { semantic_content_integrity_trap();return; }
    const uint64_t leaf_count=(bytes+MODEL_STREAM_LEAF_BYTES-1)/MODEL_STREAM_LEAF_BYTES;
    if(!inputs.baseline_leaf_scratch || !inputs.candidate_leaf_scratch || !inputs.position_scratch ||
       leaf_count>inputs.leaf_capacity) { semantic_content_integrity_trap();return; }
    auto& output=state.rows[state.canary_index];
    const auto* positions=reinterpret_cast<const ModelCanaryStreamPosition*>(inputs.position_scratch);
    for(uint64_t local=0;local<inputs.count;++local) {
        const auto& partial=positions[local];
        semantic_stream_merge_native(work,partial.work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(output.reason)continue;
        if(partial.metrics.reason) { output.reason=partial.metrics.reason;continue; }
        output.obligations+=partial.metrics.obligations;
        output.baseline_correct+=partial.metrics.baseline_correct;
        output.candidate_correct+=partial.metrics.candidate_correct;output.lost+=partial.metrics.lost;
        output.retention_denominator+=partial.metrics.retention_denominator;
        output.protected_seen+=partial.metrics.protected_seen;output.protected_lost+=partial.metrics.protected_lost;
        output.positions+=partial.metrics.positions;
        if(__longlong_as_double((long long)partial.metrics.maximum_bits)>
           __longlong_as_double((long long)output.maximum_bits))output.maximum_bits=partial.metrics.maximum_bits;
    }
    const auto* baseline_leaves=reinterpret_cast<const ModelStreamLeaf*>(inputs.baseline_leaf_scratch);
    const auto* candidate_leaves=reinterpret_cast<const ModelStreamLeaf*>(inputs.candidate_leaf_scratch);
    for(uint64_t leaf=0;leaf<leaf_count;++leaf) {
        semantic_stream_merge_native(work,baseline_leaves[leaf].work);
        semantic_stream_merge_native(work,candidate_leaves[leaf].work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,2);
    }
    semantic_stream_fold_block(baseline_leaves,leaf_count,bytes,output.row_ordinal,
        inputs.position_begin,inputs.count,output.baseline_digest,work);
    semantic_stream_fold_block(candidate_leaves,leaf_count,candidate_bytes,output.row_ordinal,
        inputs.position_begin,inputs.count,output.candidate_digest,work);
    state.position+=inputs.count;
    if(state.position==state.capacity) {
        const auto* baseline_rows=reinterpret_cast<const uint64_t*>(inputs.baseline_row_seals);
        const auto* candidate_rows=reinterpret_cast<const uint64_t*>(inputs.candidate_row_seals);
        if(!publication_identity_equal(output.baseline_digest,baseline_rows+output.row_ordinal*4,work) ||
           !publication_identity_equal(output.candidate_digest,candidate_rows+output.row_ordinal*4,work)) {
            semantic_content_integrity_trap();return;
        }
        state.position=0;++state.canary_index;
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(output)+16);
}

extern "C" __global__ void semantic_model_canary_stream_finish(ModelCanaryStreamInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    auto& state=*reinterpret_cast<ModelCanaryStreamState*>(inputs.state);
    auto* work=&state.work;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    if(state.abi!=1 || state.phase!=1 || state.canary_index!=5 || state.position) {
        semantic_content_integrity_trap();return;
    }
    const auto* canaries=reinterpret_cast<const SemanticTrainingCanaryRecord*>(inputs.canaries);
    for(uint64_t item=0;item<5;++item) {
        auto& row=state.rows[item];const auto& canary=canaries[item];
        if(!row.reason && ((canary.kind==1 || canary.kind==4) &&
           (!row.obligations || row.obligations!=canary.obligation_member_count)))row.reason=7;
        if(!row.reason && canary.kind==2 && (!row.retention_denominator ||
           row.protected_seen!=canary.protected_member_count))row.reason=5;
        if(!row.reason && canary.kind==3 && !row.positions)row.reason=5;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    }
    const auto* forwards=reinterpret_cast<const ModelForwardReceipt*>(inputs.forward_receipts);
    ModelCanaryStreamReceipt receipt{};receipt.abi=1;receipt.availability=2;
    receipt.row_count=state.row_count;receipt.capacity=state.capacity;receipt.vocabulary=state.vocabulary;
    semantic_stream_copy_identity(receipt.baseline_identity,forwards[0].identity,work);
    semantic_stream_copy_identity(receipt.candidate_identity,forwards[1].identity,work);
    semantic_stream_copy_identity(receipt.selection_identity,state.selection_identity,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(state.rows),sizeof(state.rows),receipt.summary_digest,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipt),
        sizeof(receipt)-sizeof(receipt.identity),receipt.identity,work);
    *reinterpret_cast<ModelCanaryStreamReceipt*>(inputs.receipt)=receipt;state.phase=2;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(receipt)+8);
}

// Called once by the existing whole-Update canary gate, after all three stream
// finishes. These are native checks, not producer-side verdicts or extra model
// invocations. The gate still owns resource canary, bounds, refusal precedence,
// final suffix accounting and the sole joint publication.
__device__ bool semantic_stream_comparison_valid(const ModelCanaryStreamState& state,
        const ModelCanaryStreamReceipt& receipt,const ModelForwardReceipt* forwards,
        const SemanticTrainingViewSelection& selection,const SemanticTrainingCanaryRecord* canaries,
        uint64_t rows,uint64_t capacity,uint64_t vocabulary,semantic_graph::NativeWorkTally* work) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(state)+sizeof(receipt)+2*sizeof(ModelForwardReceipt));
    if(state.abi!=1 || state.phase!=2 || state.canary_index!=5 || state.position ||
       state.row_count!=rows || state.capacity!=capacity || state.vocabulary!=vocabulary ||
       receipt.abi!=1 || receipt.availability!=2 || receipt.row_count!=rows ||
       receipt.capacity!=capacity || receipt.vocabulary!=vocabulary ||
       !publication_identity_equal(state.selection_identity,selection.identity,work) ||
       !publication_identity_equal(receipt.selection_identity,selection.identity,work) ||
       !publication_identity_equal(receipt.baseline_identity,forwards[0].identity,work) ||
       !publication_identity_equal(receipt.candidate_identity,forwards[1].identity,work))return false;
    for(uint64_t item=0;item<5;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(state.rows[item].kind!=canaries[item].kind ||
           state.rows[item].row_ordinal!=canaries[item].row_ordinal ||
           !publication_identity_equal(state.rows[item].identity,canaries[item].identity,work))return false;
    }
    uint64_t summary[4],identity[4];
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(state.rows),sizeof(state.rows),summary,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipt),
        sizeof(receipt)-sizeof(receipt.identity),identity,work);
    return publication_identity_equal(summary,receipt.summary_digest,work) &&
        publication_identity_equal(identity,receipt.identity,work);
}

__device__ void semantic_stream_merge_work(const ModelForwardStreamState& baseline,
        const ModelForwardStreamState& candidate,const ModelCanaryStreamState& comparison,
        ExecutionWork& work,semantic_graph::NativeWorkTally* native) {
    const semantic_graph::NativeWorkTally* sources[]={&baseline.work,&candidate.work,&comparison.work};
    for(uint32_t source=0;source<3;++source) {
        const auto& actual=*sources[source];
        uint64_t sum=0;
        for(uint32_t event=0;event<9;++event) {
            semantic_graph::charge_native(native,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(actual.events[event]>UINT64_MAX-sum) { native->overflow=work.overflow=1;return; }
            sum+=actual.events[event];
        }
        if(actual.overflow || sum!=actual.units) { native->overflow=work.overflow=1;return; }
        execution_work_merge_native(work,actual);
        for(uint32_t event=0;event<9;++event) {
            semantic_graph::charge_native(native,semantic_graph::NativeWorkEvent::TableSlot,1);
            semantic_graph::charge_native(native,static_cast<semantic_graph::NativeWorkEvent>(event),actual.events[event]);
        }
        if(native->overflow)work.overflow=1;
    }
}

__device__ bool semantic_stream_full_finite(const ModelForwardStreamState& baseline,
        const ModelForwardStreamState& candidate,semantic_graph::NativeWorkTally* work) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,2);
    if(baseline.finite>1 || candidate.finite>1) { semantic_content_integrity_trap();return false; }
    return baseline.finite && candidate.finite;
}

__device__ bool semantic_stream_canary_measurement(const ModelCanaryStreamState& state,
        uint64_t item,const SemanticTrainingCanaryRecord& canary,uint64_t& availability,
        uint64_t& reason,double& measurement) {
    if(item>=5 || canary.kind!=item+1)return false;
    const auto& row=state.rows[item];
    availability=1;reason=row.reason;measurement=0.0;
    if(reason)return true;
    if(canary.kind==1) {
        if(!row.obligations || row.obligations!=canary.obligation_member_count) { reason=7;return true; }
        measurement=__ddiv_rn(double(int64_t(row.candidate_correct)-int64_t(row.baseline_correct)),
            double(row.obligations));availability=2;
    } else if(canary.kind==2) {
        if(!row.retention_denominator || row.protected_seen!=canary.protected_member_count) {
            reason=5;return true;
        }
        measurement=__ddiv_rn(double(int64_t(row.candidate_correct)-int64_t(row.baseline_correct)),
            double(row.retention_denominator));availability=2;
        if(row.protected_lost)reason=6;
    } else if(canary.kind==3) {
        if(!row.positions) { reason=5;return true; }
        measurement=__longlong_as_double((long long)row.maximum_bits);availability=2;
    } else if(canary.kind==4) {
        if(!row.obligations || row.obligations!=canary.obligation_member_count) { reason=7;return true; }
        measurement=-__ddiv_rn(double(row.lost),double(row.obligations));availability=2;
        if(row.lost)reason=7;
    } else {
        // Kind five remains the existing exact whole-attempt resource canary.
        return false;
    }
    return true;
}

#endif
