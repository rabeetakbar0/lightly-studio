import { describe, expect, it } from 'vitest';
import { formatBytes, progressLabel } from './useVideoTests.svelte';

describe('progressLabel', () => {
    it('says the pipeline is starting before the first video', () => {
        expect(progressLabel(null)).toBe('Starting the pipeline…');
    });

    it('names the video, its frame share and speed', () => {
        expect(
            progressLabel({
                total: 3,
                current_index: 2,
                current_video: 'clip.mp4',
                frames_done: 450,
                frames_total: 1000,
                fps: 61.2,
                pass_number: 1,
                percent: 48.3
            })
        ).toBe('Video 2 of 3 · clip.mp4 · 45% · 61 fps');
    });
});

describe('formatBytes', () => {
    it('picks a unit', () => {
        expect(formatBytes(800)).toBe('1 KB');
        expect(formatBytes(35.4 * 1024 ** 2)).toBe('35.4 MB');
        expect(formatBytes(1.2 * 1024 ** 3)).toBe('1.2 GB');
    });
});
