import '@testing-library/jest-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrainPanel from './TrainPanel.svelte';

const mocks = vi.hoisted(() => ({
    preview: vi.fn(),
    remove: vi.fn(),
    merge: vi.fn(),
    data: [] as unknown[]
}));

vi.mock('$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte', () => ({
    usePipelineDatasets: () => ({ data: mocks.data, isLoading: false, error: null }),
    usePipelineDatasetActions: () => ({
        mergeMutation: { isPending: false, isSuccess: false },
        deleteMutation: { isPending: false, isSuccess: false },
        preview: mocks.preview,
        merge: mocks.merge,
        remove: mocks.remove
    }),
    formatBytes: (bytes: number) => `${bytes} B`
}));

const dataset = (name: string) => ({
    name,
    path: `/d/${name}`,
    created_at: '2026-10-26T10:00:00+00:00',
    classes: ['GateArm', 'LedLight'],
    splits: { train2017: { images: 8, boxes: 20 }, val2017: { images: 2, boxes: 5 } },
    sources: [],
    annotation_source: 'annotation',
    size_bytes: 1024,
    images: 10,
    boxes: 25
});

describe('TrainPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.data = [];
    });

    it('shows the empty state and disables merging', () => {
        render(TrainPanel, { props: { onClose: vi.fn() } });
        expect(screen.getByText(/No datasets yet/)).toBeInTheDocument();
        expect(screen.getByTestId('merge-datasets')).toBeDisabled();
    });

    it('lists datasets and previews or deletes one', async () => {
        mocks.data = [dataset('data1'), dataset('data2')];
        render(TrainPanel, { props: { onClose: vi.fn() } });
        expect(screen.getAllByTestId('dataset-row')).toHaveLength(2);
        expect(screen.getAllByText(/10 images · 25 boxes · train 8 · val 2/)).toHaveLength(2);
        await fireEvent.click(screen.getByRole('button', { name: 'Preview data1' }));
        expect(mocks.preview).toHaveBeenCalledWith('data1');
        await fireEvent.click(screen.getByRole('button', { name: 'Delete data2' }));
        await waitFor(() => expect(screen.getByTestId('confirm-delete')).toBeInTheDocument());
        await fireEvent.click(screen.getByTestId('confirm-delete'));
        expect(mocks.remove).toHaveBeenCalledWith('data2');
    });
});
