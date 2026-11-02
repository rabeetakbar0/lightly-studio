import '@testing-library/jest-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import TrainPanel from './TrainPanel.svelte';

const mocks = vi.hoisted(() => ({
    preview: vi.fn(),
    remove: vi.fn(),
    merge: vi.fn(),
    start: vi.fn(),
    buildTest: vi.fn(),
    data: [] as unknown[],
    runs: [] as unknown[]
}));

vi.mock('$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte', () => ({
    usePipelineDatasets: () => ({ data: mocks.data, isLoading: false, error: null }),
    usePipelineDatasetActions: () => ({
        mergeMutation: { isPending: false, isSuccess: false },
        buildTestMutation: { isPending: false, isSuccess: false },
        buildTest: mocks.buildTest,
        deleteMutation: { isPending: false, isSuccess: false },
        preview: mocks.preview,
        merge: mocks.merge,
        remove: mocks.remove
    }),
    formatBytes: (bytes: number) => `${bytes} B`,
    splitSummary: () => ''
}));

vi.mock('$lib/hooks/usePipelineTraining/usePipelineTraining.svelte', () => ({
    useTrainingOptions: () => ({
        data: {
            defaults: { batch_size: 16, fp16: true },
            presets: { m: [0.67, 0.75] },
            baselines: [],
            environment: { yolox_dir: '/y', python: 'py', weights: [], problems: [], ready: true }
        }
    }),
    useTrainingRuns: () => ({ data: mocks.runs, isLoading: false, error: null }),
    useTrainingLog: () => ({ data: { text: '' }, isLoading: false, error: null }),
    useTestDatasets: () => ({ data: [], isLoading: false, isSuccess: true, error: null }),
    useTrainingActions: () => ({
        startMutation: { isPending: false, isSuccess: false },
        testMutation: { isPending: false, isSuccess: false },
        stopMutation: { isPending: false, isSuccess: false },
        deleteMutation: { isPending: false, isSuccess: false },
        start: mocks.start,
        startTest: vi.fn(),
        stop: vi.fn(),
        remove: vi.fn()
    }),
    isActiveRun: (run: { status: string }) => run.status === 'running',
    testResultLabel: () => 'AP 76.3'
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
        expect(screen.getByTestId('train-model')).toBeDisabled();
        expect(screen.getByTestId('build-test-dataset')).toBeDisabled();
        expect(screen.getByText(/No runs yet/)).toBeInTheDocument();
    });

    it('builds a test dataset from chosen datasets', async () => {
        mocks.data = [dataset('data1'), dataset('data2')];
        render(TrainPanel, { props: { onClose: vi.fn() } });
        await fireEvent.click(screen.getByTestId('build-test-dataset'));
        await waitFor(() => expect(screen.getByTestId('build-test-confirm')).toBeInTheDocument());
        expect(screen.getByTestId('build-test-confirm')).toBeDisabled();
        await fireEvent.click(screen.getByLabelText('Draw from data1'));
        await fireEvent.click(screen.getByLabelText('Draw from data2'));
        await fireEvent.input(screen.getByTestId('build-test-amount'), {
            target: { value: '0.2' }
        });
        await fireEvent.input(screen.getByTestId('build-test-name'), {
            target: { value: 'gate_test' }
        });
        await fireEvent.click(screen.getByTestId('build-test-confirm'));
        expect(mocks.buildTest).toHaveBeenCalledWith({
            name: 'gate_test',
            sources: ['data1', 'data2'],
            fraction: 0.2,
            count: null,
            split_strategy: 'embedding',
            groups: null
        });
    });

    it('opens the train dialog and lists runs', async () => {
        mocks.data = [dataset('data1')];
        mocks.runs = [
            {
                name: 'run1',
                dataset: 'data1',
                classes: ['GateArm'],
                config: {},
                status: 'finished',
                created_at: '2026-10-27T10:00:00+00:00',
                finished_at: null,
                exit_code: 0,
                onnx_path: '/models/run1.onnx',
                error: null,
                kind: 'train',
                model: null,
                test_result: null,
                progress: {
                    epoch: 2,
                    max_epoch: 2,
                    iteration: 0,
                    max_iteration: 0,
                    eta: '',
                    best_ap50_95: null,
                    evaluations: [],
                    last_line: ''
                }
            }
        ];
        render(TrainPanel, { props: { onClose: vi.fn() } });
        expect(screen.getAllByTestId('run-row')).toHaveLength(1);
        expect(screen.getByText('Exported: /models/run1.onnx')).toBeInTheDocument();
        await fireEvent.click(screen.getByTestId('train-model'));
        await waitFor(() => expect(screen.getByTestId('train-confirm')).toBeInTheDocument());
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
