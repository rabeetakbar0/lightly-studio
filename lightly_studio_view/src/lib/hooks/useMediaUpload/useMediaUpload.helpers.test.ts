import { describe, expect, it } from 'vitest';
import { apiErrorMessage, formatDuration, splitMediaFiles } from './useMediaUpload.helpers';

const file = (name: string, type = ''): File => new File(['x'], name, { type });

describe('splitMediaFiles', () => {
    it('sorts by MIME type and falls back to the extension', () => {
        const result = splitMediaFiles([
            file('a.png', 'image/png'),
            file('b.MOV'),
            file('c.mp4', 'video/mp4'),
            file('notes.txt', 'text/plain'),
            file('d.jpeg')
        ]);
        expect(result.images.map((f) => f.name)).toEqual(['a.png', 'd.jpeg']);
        expect(result.videos.map((f) => f.name)).toEqual(['b.MOV', 'c.mp4']);
        expect(result.unsupported.map((f) => f.name)).toEqual(['notes.txt']);
    });
});

describe('formatDuration', () => {
    it('formats minutes and hours', () => {
        expect(formatDuration(0)).toBe('0:00');
        expect(formatDuration(65.4)).toBe('1:05');
        expect(formatDuration(3725)).toBe('1:02:05');
    });
});

describe('apiErrorMessage', () => {
    it('reads error or detail and falls back', () => {
        expect(apiErrorMessage({ error: 'bad' }, 'fallback')).toBe('bad');
        expect(apiErrorMessage({ detail: 'missing' }, 'fallback')).toBe('missing');
        expect(apiErrorMessage(new Error('x'), 'fallback')).toBe('fallback');
    });
});
