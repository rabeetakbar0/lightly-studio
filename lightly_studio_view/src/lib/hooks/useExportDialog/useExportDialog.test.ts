import { get } from 'svelte/store';
import { beforeEach, describe, expect, it } from 'vitest';
import { useExportDialog } from './useExportDialog';

describe('useExportDialog', () => {
    beforeEach(() => {
        // Reset shared store state between tests by closing any open dialog
        useExportDialog().closeExportDialog();
    });

    it('opens the dialog', () => {
        const { isExportDialogOpen, openExportDialog } = useExportDialog();

        openExportDialog();

        expect(get(isExportDialogOpen)).toBe(true);
    });

    it('closes the dialog', () => {
        const { isExportDialogOpen, openExportDialog, closeExportDialog } = useExportDialog();
        openExportDialog();

        closeExportDialog();

        expect(get(isExportDialogOpen)).toBe(false);
    });

    it('stays closed when closing an already closed dialog', () => {
        const { isExportDialogOpen, closeExportDialog } = useExportDialog();

        closeExportDialog();

        expect(get(isExportDialogOpen)).toBe(false);
    });
});
