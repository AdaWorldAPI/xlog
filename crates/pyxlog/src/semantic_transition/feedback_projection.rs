//! Pure feedback metadata of the same original model-content owner.

use super::*;

impl ContentStepOwner {
    pub(super) fn feedback_intervention(
        &self,
        py: Python<'_>,
        session: &PySemanticTransitionSession,
        planning: bool,
    ) -> PyResult<Option<Py<PyTuple>>> {
        #[cfg(not(feature = "semantic-policy"))]
        let _ = planning;
        session.require_creator()?;
        {
            let owner = session.owner()?;
            self.binding(py, &owner, false)?;
        }
        if !session.learning_preparing.load(Ordering::Acquire) {
            return Ok(None);
        }
        #[cfg(not(feature = "semantic-policy"))]
        return Err(invalid(
            "private learning feedback requires the semantic-policy owner",
        ));
        #[cfg(feature = "semantic-policy")]
        {
            let pending = session
                .learning_transition
                .lock()
                .map_err(|_| invalid("learning-phase retention mutex is poisoned"))?
                .as_ref()
                .map(|pending| pending.clone_ref(py))
                .ok_or_else(|| {
                    invalid("private model content lost its original learning-feedback owner")
                })?;
            let pending = pending.borrow(py);
            if pending.private_numeric_active() {
                return pending
                    .private_numeric_feedback_projection(py, session, self)
                    .map(Some);
            }
            let ContentStepOwner::Published(parent) = self else {
                return Err(invalid("source feedback projection requires its original acquired source, not a private prepared step"));
            };
            if let Some(projection) =
                pending.private_evaluation_feedback_projection(py, session, parent, planning)?
            {
                return Ok(Some(projection));
            }
            if pending.is_original_source(py, session) {
                pending.require_source_feedback_projection(py, session, parent, planning)?;
                Ok(None)
            } else {
                pending
                    .private_restore_feedback_projection(py, session, parent)
                    .map(Some)
            }
        }
    }
}

#[pymethods]
impl PySemanticTransitionTaskUse {
    /// Same original phase metadata, without creating a numerical witness.
    fn learning_feedback_intervention(
        &self,
        py: Python<'_>,
        parent: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyTuple>>> {
        let parent = ContentStepOwner::from_python(parent)?;
        match &parent {
            ContentStepOwner::Published(parent) => parent.borrow(py).require_task(py, self)?,
            ContentStepOwner::Prepared(step) => step.borrow(py).require_task(py, self)?,
        }
        let session = self.session.borrow(py);
        parent.feedback_intervention(py, &session, true)
    }
}
