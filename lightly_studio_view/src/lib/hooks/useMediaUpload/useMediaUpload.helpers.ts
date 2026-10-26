const IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tif', 'tiff']);
const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'm4v']);

export interface FrameSettings {
    /** Length in seconds of the densely sampled window at the start and at the end. */
    edgeWindowSeconds: number;
    /** Seconds between frames inside the dense windows. */
    edgeIntervalSeconds: number;
    /** Number of evenly spaced frames taken from the middle section. */
    middleFrames: number;
    /** Upper bound on the frames per video. */
    maxFrames: number;
}

export const defaultFrameSettings: FrameSettings = {
    edgeWindowSeconds: 20,
    edgeIntervalSeconds: 1,
    middleFrames: 35,
    maxFrames: 100
};

interface SplitMediaFiles {
    images: File[];
    videos: File[];
    unsupported: File[];
}

const extensionOf = (file: File): string => file.name.split('.').pop()?.toLowerCase() ?? '';

/** The file name without its extension, the default tag for a video's frames. */
export const fileStem = (file: File): string => file.name.replace(/\.[^.]+$/, '');

/** Sorts dropped files into images, videos and everything else, by MIME type or extension. */
export function splitMediaFiles(files: File[]): SplitMediaFiles {
    const result: SplitMediaFiles = { images: [], videos: [], unsupported: [] };
    for (const file of files) {
        const extension = extensionOf(file);
        if (file.type.startsWith('image/') || IMAGE_EXTENSIONS.has(extension)) {
            result.images.push(file);
        } else if (file.type.startsWith('video/') || VIDEO_EXTENSIONS.has(extension)) {
            result.videos.push(file);
        } else {
            result.unsupported.push(file);
        }
    }
    return result;
}

/** Reads a video's duration in seconds through a detached video element; rejects when unreadable. */
export function readVideoDuration(file: File): Promise<number> {
    return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(file);
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(url);
            resolve(Number.isFinite(video.duration) ? video.duration : 0);
        };
        video.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error(`Cannot read ${file.name}`));
        };
        video.src = url;
    });
}

/** Formats seconds as m:ss, or h:mm:ss above an hour. */
export function formatDuration(totalSeconds: number): string {
    const seconds = Math.round(totalSeconds);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    const mm = h > 0 ? String(m).padStart(2, '0') : String(m);
    return `${h > 0 ? `${h}:` : ''}${mm}:${String(s).padStart(2, '0')}`;
}

/** Extracts the message from an API error body, which uses `error` or `detail`. */
export function apiErrorMessage(error: unknown, fallback: string): string {
    const body = error as { error?: unknown; detail?: unknown } | null;
    if (typeof body?.error === 'string') return body.error;
    if (typeof body?.detail === 'string') return body.detail;
    return fallback;
}
