import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import RunRow from './RunRow.svelte';
import type { TrainingRunView } from '$lib/api/lightly_studio_local';

const run = (overrides: Partial<TrainingRunView>): TrainingRunView =>
    ({
        name: 'run1',
        dataset: 'data1',
        classes: ['GateArm', 'LedLight'],
        config: {},
        status: 'running',
        created_at: '2026-10-27T10:00:00+00:00',
        finished_at: null,
        exit_code: null,
        onnx_path: null,
        error: null,
        kind: 'train',
        model: null,
        test_result: null,
        progress: {
            epoch: 12,
            max_epoch: 150,
            iteration: 20,
            max_iteration: 557,
            eta: '9:30:00',
            best_ap50_95: 0.8211,
            evaluations: [],
            last_line: ''
        },
        ...overrides
    }) as TrainingRunView;

describe('RunRow', () => {
    it('shows progress of an active run and offers stop', async () => {
        const onStop = vi.fn();
        render(RunRow, { props: { run: run({}), onLog: vi.fn(), onStop, onDelete: vi.fn() } });
        expect(screen.getByText('running')).toBeInTheDocument();
        expect(
            screen.getByText('epoch 12/150 · iter 20/557 · ETA 9:30:00 · best AP 82.1')
        ).toBeInTheDocument();
        await fireEvent.click(screen.getByRole('button', { name: 'Stop run1' }));
        expect(onStop).toHaveBeenCalled();
        expect(screen.queryByRole('button', { name: 'Delete run1' })).not.toBeInTheDocument();
    });

    it('offers to test a finished training and shows a test run result', async () => {
        const onTest = vi.fn();
        render(RunRow, {
            props: {
                run: run({ status: 'finished', onnx_path: '/m/run1.onnx' }),
                onLog: vi.fn(),
                onStop: vi.fn(),
                onDelete: vi.fn(),
                onTest
            }
        });
        await fireEvent.click(screen.getByRole('button', { name: 'Test run1' }));
        expect(onTest).toHaveBeenCalled();

        render(RunRow, {
            props: {
                run: run({
                    name: 'run1__test__t1__x',
                    kind: 'test',
                    model: 'run1',
                    dataset: 't1',
                    status: 'finished',
                    test_result: {
                        dataset: 't1',
                        images: 120,
                        ap50_95: 0.763,
                        ap50: 0.98,
                        per_class_ap: { GateArm: 0.8 },
                        overlap_images: 3,
                        run_name: 'run1__test__t1__x',
                        created_at: ''
                    }
                }),
                onLog: vi.fn(),
                onStop: vi.fn(),
                onDelete: vi.fn()
            }
        });
        expect(screen.getByText('run1 on t1')).toBeInTheDocument();
        expect(screen.getByText('test')).toBeInTheDocument();
        expect(
            screen.getByText('AP 76.3 · AP50 98.0 · 120 images · 3 seen in training')
        ).toBeInTheDocument();
    });

    it('shows the error and offers delete for a finished run', async () => {
        const onDelete = vi.fn();
        render(RunRow, {
            props: {
                run: run({ status: 'failed', error: 'Training exited with code 1; see the log.' }),
                onLog: vi.fn(),
                onStop: vi.fn(),
                onDelete
            }
        });
        expect(screen.getByText('failed')).toBeInTheDocument();
        expect(screen.getByText('Training exited with code 1; see the log.')).toBeInTheDocument();
        expect(screen.getByText('epoch 12/150 · best AP 82.1')).toBeInTheDocument();
        await fireEvent.click(screen.getByRole('button', { name: 'Delete run1' }));
        expect(onDelete).toHaveBeenCalled();
    });
});
