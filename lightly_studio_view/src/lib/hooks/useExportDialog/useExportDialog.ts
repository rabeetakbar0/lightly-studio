import { get, writable } from 'svelte/store';

const isExportDialogOpen = writable(false);

export function useExportDialog() {
    const openExportDialog = () => {
        if (get(isExportDialogOpen)) return;
        isExportDialogOpen.set(true);
    };

    const closeExportDialog = () => {
        if (!get(isExportDialogOpen)) return;
        isExportDialogOpen.set(false);
    };

    return {
        isExportDialogOpen,
        openExportDialog,
        closeExportDialog
    };
}
