import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import PrepareDatasetTab from './PrepareDatasetTab.svelte';

const mocks = vi.hoisted(() => ({ mutate: vi.fn() }));

vi.mock('$app/state', () => ({ page: { params: { collection_id: 'c1' } } }));
vi.mock('$lib/hooks', () => ({
    useImageFilters: () => ({
        imageFilter: { subscribe: (fn: (v: null) => void) => (fn(null), () => {}) }
    })
}));
vi.mock('$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte', () => ({
    usePipelineDatasetActions: () => ({
        exportMutation: { isPending: false, data: null, mutate: mocks.mutate }
    }),
    splitSummary: () => ''
}));

describe('PrepareDatasetTab', () => {
    it('sends both fractions, the spread and the groups', async () => {
        render(PrepareDatasetTab, {
            props: {
                annotationSources: [{ id: 'src', name: 'annotation' }],
                selectedAnnotationCollectionId: 'src'
            }
        });
        await fireEvent.input(screen.getByTestId('prepare-dataset-name'), {
            target: { value: 'data1' }
        });
        await fireEvent.input(screen.getByTestId('prepare-test-fraction'), {
            target: { value: '0.2' }
        });
        await fireEvent.input(screen.getByTestId('prepare-groups'), { target: { value: '8' } });
        await fireEvent.click(screen.getByTestId('submit-button-prepare-dataset'));
        expect(mocks.mutate).toHaveBeenCalledWith({
            path: { collection_id: 'c1' },
            body: {
                name: 'data1',
                annotation_collection_id: 'src',
                image_filter: null,
                val_fraction: 0.1,
                test_fraction: 0.2,
                split_by: 'tag',
                split_strategy: 'embedding',
                groups: 8
            }
        });
    });

    it('rejects fractions that leave no training images', async () => {
        render(PrepareDatasetTab, {
            props: {
                annotationSources: [{ id: 'src', name: 'annotation' }],
                selectedAnnotationCollectionId: 'src'
            }
        });
        await fireEvent.input(screen.getByTestId('prepare-dataset-name'), {
            target: { value: 'data1' }
        });
        await fireEvent.input(screen.getByTestId('prepare-val-fraction'), {
            target: { value: '0.5' }
        });
        await fireEvent.input(screen.getByTestId('prepare-test-fraction'), {
            target: { value: '0.5' }
        });
        expect(screen.getByText(/together must stay below 1/)).toBeInTheDocument();
        expect(screen.getByTestId('submit-button-prepare-dataset')).toBeDisabled();
    });
});
