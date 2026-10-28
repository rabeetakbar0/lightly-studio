import { writable } from 'svelte/store';

// Module-level store is intentional: multiple invocations of useSamplingDialog()
// (e.g. from SamplingCombinationDialog and useSamplingCombinationDialog) must share
// the same state.
const isSamplingDialogOpen = writable(false);

export function useSamplingDialog() {
    const openSamplingDialog = () => isSamplingDialogOpen.set(true);

    const closeSamplingDialog = () => isSamplingDialogOpen.set(false);

    return {
        isSamplingDialogOpen,
        openSamplingDialog,
        closeSamplingDialog
    };
}
