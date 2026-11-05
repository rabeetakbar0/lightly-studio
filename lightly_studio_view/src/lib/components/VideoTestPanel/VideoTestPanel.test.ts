import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import VideoTestPanel from './VideoTestPanel.svelte';

const mocks = vi.hoisted(() => ({
    start: vi.fn(),
    stop: vi.fn(),
    ready: true,
    runs: [] as unknown[]
}));

vi.mock('$lib/hooks/useVideoTests/useVideoTests.svelte', async (importOriginal) => ({
    ...(await importOriginal<object>()),
    useVideoTestOptions: () => ({
        data: {
            ready: mocks.ready,
            problems: mocks.ready ? [] : ['LIGHTLY_STUDIO_TESTAPP_DIR is not set.'],
            testapp_dir: '/t',
            uploads_dir: '/t/uploads',
            results_dir: '/t/uploads/results',
            models: [{ name: 'yolox_m_slm_v4', classes: ['GateArm', 'LedLight'] }]
        },
        isLoading: false
    }),
    useVideoTests: () => ({ data: mocks.runs, isLoading: false, error: null }),
    useVideoTestActions: () => ({
        startMutation: { isPending: false, isSuccess: false },
        start: mocks.start,
        stop: mocks.stop
    })
}));

const video = (name: string) => new File(['v'], name, { type: 'video/mp4' });

describe('VideoTestPanel', () => {
    beforeEach(() => {
        mocks.start.mockReset();
        mocks.ready = true;
        mocks.runs = [];
    });

    it('runs the chosen videos with the default model', async () => {
        render(VideoTestPanel, { props: { onClose: vi.fn() } });
        const run = screen.getByTestId('video-test-run');
        expect(run).toBeDisabled();

        const input = screen.getByTestId('video-test-input');
        await fireEvent.change(input, {
            target: { files: [video('a.mp4'), video('b.MOV'), video('notes.txt')] }
        });
        expect(screen.getByTestId('video-test-files')).toHaveTextContent('a.mp4');
        expect(screen.getByTestId('video-test-files')).toHaveTextContent('b.MOV');
        expect(screen.getByTestId('video-test-files')).not.toHaveTextContent('notes.txt');
        expect(run).toHaveTextContent('Run on 2 videos');

        await fireEvent.click(screen.getByLabelText('Remove b.MOV'));
        await fireEvent.click(run);
        expect(mocks.start).toHaveBeenCalledWith([expect.any(File)], 'yolox_m_slm_v4');
        expect(mocks.start.mock.calls[0][0][0].name).toBe('a.mp4');
    });

    it('takes dropped videos', async () => {
        render(VideoTestPanel, { props: { onClose: vi.fn() } });
        await fireEvent.drop(screen.getByTestId('video-test-panel'), {
            dataTransfer: { types: ['Files'], files: [video('drop.mp4')] }
        });
        expect(screen.getByTestId('video-test-files')).toHaveTextContent('drop.mp4');
    });

    it('lists the setup problems when the TestApp is not configured', () => {
        mocks.ready = false;
        render(VideoTestPanel, { props: { onClose: vi.fn() } });
        expect(screen.getByTestId('video-test-problems')).toHaveTextContent(
            'LIGHTLY_STUDIO_TESTAPP_DIR is not set.'
        );
        expect(screen.queryByTestId('video-test-run')).not.toBeInTheDocument();
    });
});
