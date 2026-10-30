import '@testing-library/jest-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ModelsPanel from './ModelsPanel.svelte';

const mocks = vi.hoisted(() => ({
    start: vi.fn(),
    models: [] as unknown[]
}));

vi.mock('$lib/hooks/usePipelineModels/usePipelineModels.svelte', async (importOriginal) => ({
    ...(await importOriginal<object>()),
    usePipelineModels: () => ({ data: mocks.models, isLoading: false, error: null })
}));
vi.mock('$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte', () => ({
    usePipelineDatasets: () => ({ data: [{ name: 'data1' }], isLoading: false, error: null }),
    formatBytes: (bytes: number) => `${bytes} B`
}));
vi.mock('$lib/hooks/usePipelineTraining/usePipelineTraining.svelte', () => ({
    useTrainingOptions: () => ({
        data: {
            defaults: { depth: 0.67, width: 0.75, batch_size: 16, baseline: 'weights:yolox_m.pth' },
            presets: { m: [0.67, 0.75] },
            baselines: [
                {
                    value: 'model:v4',
                    label: 'v4',
                    group: 'Trained models',
                    depth: 0.67,
                    width: 0.75,
                    classes: [],
                    available: true
                }
            ],
            environment: { yolox_dir: '/y', python: 'py', weights: [], problems: [], ready: true }
        }
    }),
    useTrainingActions: () => ({
        startMutation: { isPending: false, isSuccess: false },
        start: mocks.start
    })
}));

const model = (name: string, card: object | null) => ({
    name,
    classes: ['GateArm', 'LedLight'],
    onnx_path: `/models/${name}.onnx`,
    has_checkpoint: true,
    size_bytes: 2048,
    modified_at: '2026-10-27T10:00:00+00:00',
    depth: 0.67,
    width: 0.75,
    card
});
const card = (dataset: string, epochs: number) => ({
    run_name: 'r',
    dataset: {
        name: dataset,
        classes: ['GateArm', 'LedLight'],
        images: 100,
        boxes: 300,
        splits: {},
        annotation_source: 'annotation',
        sources: []
    },
    baseline: 'weights:yolox_m.pth',
    config: { depth: 0.67, width: 0.75, batch_size: 16, max_epoch: epochs },
    best_ap50_95: 0.82,
    evaluations: [],
    trained_epochs: epochs,
    created_at: '2026-10-27T10:00:00+00:00',
    finished_at: '2026-10-27T12:00:00+00:00'
});

describe('ModelsPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mocks.models = [];
    });

    it('shows the empty state', () => {
        render(ModelsPanel, { props: { onClose: vi.fn() } });
        expect(screen.getByText(/No models yet/)).toBeInTheDocument();
        expect(screen.getByTestId('compare-models')).toBeDisabled();
    });

    it('lists models with their history and compares two of them', async () => {
        mocks.models = [
            model('v4', null),
            model('v5', card('data1', 150)),
            model('v6', card('data2', 60))
        ];
        render(ModelsPanel, { props: { onClose: vi.fn() } });
        expect(screen.getAllByTestId('model-row')).toHaveLength(3);
        expect(screen.getByText(/No training history/)).toBeInTheDocument();
        expect(
            screen.getByText(/Trained on data1 \(100 images, 300 boxes, from "annotation"\)/)
        ).toBeInTheDocument();

        await fireEvent.click(screen.getByLabelText('Select v5 for comparison'));
        await fireEvent.click(screen.getByLabelText('Select v6 for comparison'));
        await fireEvent.click(screen.getByTestId('compare-models'));
        await waitFor(() => expect(screen.getByTestId('compare-table')).toBeInTheDocument());
        const differing = screen
            .getAllByRole('row')
            .filter((r) => r.getAttribute('data-differs') === 'true')
            .map((r) => Array.from(r.querySelectorAll('td')).map((td) => td.textContent?.trim()));
        expect(differing).toEqual([
            ['Dataset · Name', 'data1', 'data2'],
            ['Result · Epochs trained', '150', '60'],
            ['Schedule · Epochs', '150', '60']
        ]);
    });

    it('opens the train dialog from a model with its settings as the start', async () => {
        mocks.models = [model('v5', card('data1', 150))];
        render(ModelsPanel, { props: { onClose: vi.fn() } });
        await fireEvent.click(screen.getByRole('button', { name: 'Train from v5' }));
        await waitFor(() => expect(screen.getByTestId('train-confirm')).toBeInTheDocument());
        await fireEvent.input(screen.getByTestId('train-name'), { target: { value: 'v7' } });
        await fireEvent.click(screen.getByTestId('train-confirm'));
        expect(mocks.start).toHaveBeenCalledWith({
            name: 'v7',
            dataset: 'data1',
            config: expect.objectContaining({ baseline: 'model:v5', max_epoch: 150 })
        });
    });
});
