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
