// App.jsx autosaves the WHOLE settings object after any edit. If a successful recognition apply
// did not also land in that object, the next unrelated edit (a slider, a toggle) would post the old
// recognition_model / recognition_provider back and silently undo the saved choice.
export const mergeSelection = (settings, selection) => ({
  ...(settings || {}),
  recognition_model: selection.model,
  recognition_provider: selection.provider,
});
