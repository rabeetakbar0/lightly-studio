import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import ConfirmDeleteDialog from './ConfirmDeleteDialog.svelte';

const defaultProps = {
    title: 'Delete tag and its images?',
    description: 'Every image carrying "clip" is removed.',
    onConfirm: vi.fn(),
    onCancel: vi.fn()
};

describe('ConfirmDeleteDialog', () => {
    it('shows the text and calls the callbacks', async () => {
        render(ConfirmDeleteDialog, { props: defaultProps });
        expect(screen.getByText('Delete tag and its images?')).toBeInTheDocument();
        expect(screen.getByText(/This cannot be undone/)).toBeInTheDocument();
        await fireEvent.click(screen.getByTestId('confirm-delete'));
        expect(defaultProps.onConfirm).toHaveBeenCalled();
        await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(defaultProps.onCancel).toHaveBeenCalled();
    });

    it('blocks cancel while pending', async () => {
        const onCancel = vi.fn();
        render(ConfirmDeleteDialog, { props: { ...defaultProps, isPending: true, onCancel } });
        expect(screen.getByTestId('confirm-delete')).toHaveTextContent('Deleting…');
        await fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(onCancel).not.toHaveBeenCalled();
    });
});
