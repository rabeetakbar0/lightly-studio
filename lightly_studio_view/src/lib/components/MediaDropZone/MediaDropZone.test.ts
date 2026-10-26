import '@testing-library/jest-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import MediaDropZone from './MediaDropZone.svelte';

const mocks = vi.hoisted(() => ({
    uploadImages: vi.fn(),
    uploadVideos: vi.fn(),
    planFrames: vi.fn().mockResolvedValue(7),
    toastWarning: vi.fn()
}));

vi.mock('$lib/hooks/useMediaUpload/useMediaUpload.svelte', () => ({
    useMediaUpload: () => ({
        uploadImages: mocks.uploadImages,
        uploadVideos: mocks.uploadVideos,
        planFrames: mocks.planFrames,
        isPending: false
    })
}));
vi.mock('svelte-sonner', () => ({
    toast: { warning: mocks.toastWarning, success: vi.fn(), error: vi.fn(), loading: vi.fn() }
}));

const dragEvent = (type: string, files: File[] = []) => {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperty(event, 'dataTransfer', { value: { types: ['Files'], files } });
    return event;
};

describe('MediaDropZone', () => {
    beforeEach(() => vi.clearAllMocks());

    it('shows the overlay while files are dragged over the window', async () => {
        render(MediaDropZone, { props: { collectionId: 'c1' } });
        expect(screen.queryByTestId('media-drop-overlay')).not.toBeInTheDocument();
        await fireEvent(window, dragEvent('dragenter'));
        expect(screen.getByTestId('media-drop-overlay')).toBeInTheDocument();
        await fireEvent(window, dragEvent('dragleave'));
        expect(screen.queryByTestId('media-drop-overlay')).not.toBeInTheDocument();
    });

    it('shows a standing drop target with a file picker when the collection is empty', async () => {
        render(MediaDropZone, { props: { collectionId: 'c1', isEmpty: true } });
        expect(screen.getByTestId('media-drop-empty')).toHaveTextContent(
            'This collection is empty'
        );
        const input = screen.getByTestId('media-file-input') as HTMLInputElement;
        const image = new File(['x'], 'a.png', { type: 'image/png' });
        Object.defineProperty(input, 'files', { value: [image], configurable: true });
        await fireEvent.change(input);
        await waitFor(() => expect(screen.getByTestId('image-tag-input')).toBeInTheDocument());
        expect(mocks.uploadImages).not.toHaveBeenCalled();
    });

    it('asks for a tag for dropped images and for frame settings for videos', async () => {
        render(MediaDropZone, { props: { collectionId: 'c1' } });
        const image = new File(['x'], 'a.png', { type: 'image/png' });
        const video = new File(['x'], 'b.mp4', { type: 'video/mp4' });
        const other = new File(['x'], 'c.txt', { type: 'text/plain' });
        await fireEvent(window, dragEvent('drop', [image, video, other]));
        expect(mocks.toastWarning).toHaveBeenCalledWith('Skipped 1 unsupported files.');
        await waitFor(() => expect(screen.getByTestId('image-tag-input')).toBeInTheDocument());
        const tagInput = screen.getByTestId('image-tag-input') as HTMLInputElement;
        await fireEvent.input(tagInput, { target: { value: ' site-a ' } });
        await fireEvent.click(screen.getByRole('button', { name: 'Add images' }));
        expect(mocks.uploadImages).toHaveBeenCalledWith([image], 'site-a');
        await waitFor(() => expect(screen.getByTestId('video-list')).toHaveTextContent('b.mp4'));
        const videoTag = screen.getByTestId('video-tag-input') as HTMLInputElement;
        expect(videoTag.value).toBe('b');
        expect(mocks.uploadVideos).not.toHaveBeenCalled();
    });
});
