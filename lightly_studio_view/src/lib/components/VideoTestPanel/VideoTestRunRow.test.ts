import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import VideoTestRunRow from './VideoTestRunRow.svelte';
import type { VideoTestRunView } from '$lib/api/lightly_studio_local';

const run = (overrides: Partial<VideoTestRunView>): VideoTestRunView => ({
    name: '20261102-181513',
    model: 'yolox_m_slm_v4',
    model_path: '/models/yolox_m_slm_v4.onnx',
    input_dir: '/t/uploads/20261102-181513',
    output_dir: '/t/uploads/results/20261102-181513',
    status: 'running',
    created_at: '2026-11-02T18:15:13+00:00',
    finished_at: null,
    exit_code: null,
    error: null,
    videos: [
        {
            file_name: 'a.mp4',
            size_bytes: 10,
            status: 'processing',
            output_path: null,
            summary: {},
            benchmark: {}
        }
    ],
    progress: {
        total: 2,
        current_index: 1,
        current_video: 'a.mp4',
        frames_done: 300,
        frames_total: 1200,
        fps: 92.6,
        pass_number: 2,
        percent: 12.5
    },
    ...overrides
});

describe('VideoTestRunRow', () => {
    it('shows the progress of a running test and offers stop', async () => {
        const onStop = vi.fn();
        render(VideoTestRunRow, { props: { run: run({}), onLog: vi.fn(), onStop } });
        expect(screen.getByTestId('video-test-progress')).toHaveTextContent(
            '13% · Video 1 of 2 · a.mp4 · 25% · 93 fps · second pass with backup LED locations'
        );
        expect(screen.getByText('processing', { exact: false })).toBeInTheDocument();
        await fireEvent.click(screen.getByLabelText('Stop 20261102-181513'));
        expect(onStop).toHaveBeenCalled();
    });

    it('shows where each processed video was saved', () => {
        render(VideoTestRunRow, {
            props: {
                run: run({
                    status: 'finished',
                    progress: null,
                    videos: [
                        {
                            file_name: 'a.mp4',
                            size_bytes: 10,
                            status: 'done',
                            output_path: '/t/uploads/results/20261102-181513/a_processed.mp4',
                            summary: {
                                GatearmDetected: 'True',
                                GatearmDescentTime: '9.883',
                                GatearmAscentTime: '5.24',
                                TipLightBlinkCount: '0',
                                MiddleLightBlinkCount: '5',
                                BaseLightBlinkCount: '6'
                            },
                            benchmark: {}
                        },
                        {
                            file_name: 'b.mp4',
                            size_bytes: 10,
                            status: 'failed',
                            output_path: null,
                            summary: {},
                            benchmark: {}
                        }
                    ]
                }),
                onLog: vi.fn(),
                onStop: vi.fn()
            }
        });
        const [done, failed] = screen.getAllByTestId('video-test-video');
        expect(done).toHaveTextContent(
            'Processed video saved at: /t/uploads/results/20261102-181513/a_processed.mp4'
        );
        expect(done).toHaveTextContent(
            'arm detected · descent 9.9 s · ascent 5.2 s · blinks 0/5/6 (tip/middle/base)'
        );
        expect(failed).toHaveTextContent('b.mp4 · failed, see the log');
        expect(screen.queryByTestId('video-test-progress')).not.toBeInTheDocument();
        expect(screen.queryByLabelText('Stop 20261102-181513')).not.toBeInTheDocument();
    });
});
