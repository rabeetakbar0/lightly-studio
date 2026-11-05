import { createMutation, createQuery, useQueryClient } from '@tanstack/svelte-query';
import { toast } from 'svelte-sonner';
import type { VideoTestRunView } from '$lib/api/lightly_studio_local';
import {
    getVideoTestLogOptions,
    getVideoTestEnvironmentOptions,
    listVideoTestsOptions,
    startVideoTestMutation,
    stopVideoTestMutation
} from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { apiErrorMessage } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

const RUNS_KEY = [{ _id: 'listVideoTests' }];
const LOG_KEY = [{ _id: 'getVideoTestLog' }];
/** How often a running video test is re-read. */
export const POLL_INTERVAL_MS = 2000;

/** True while the pipeline runs. */
export const isRunningTest = (run: Pick<VideoTestRunView, 'status'>): boolean =>
    run.status === 'running';

/** The environment check, the upload and result folders and the models. */
export const useVideoTestOptions = () => createQuery(() => getVideoTestEnvironmentOptions());

/** All video tests, newest first; polls while one runs. */
export const useVideoTests = () =>
    createQuery(() => ({
        ...listVideoTestsOptions(),
        refetchInterval: (query) =>
            query.state.data?.some(isRunningTest) ? POLL_INTERVAL_MS : false
    }));

/** The log tail of one video test, re-read while it runs. */
export const useVideoTestLog = (getParams: () => { name: string; active: boolean }) =>
    createQuery(() => {
        const { name, active } = getParams();
        return {
            ...getVideoTestLogOptions({ path: { name }, query: { lines: 400 } }),
            refetchInterval: active ? POLL_INTERVAL_MS : false
        };
    });

/** "Video 2 of 3 · clip.mp4 · 45% · 61 fps · second pass" while a test runs. */
export const progressLabel = (progress: VideoTestRunView['progress']): string => {
    if (!progress || progress.total === 0) return 'Starting the pipeline…';
    const parts = [`Video ${progress.current_index} of ${progress.total}`];
    if (progress.current_video) parts.push(progress.current_video);
    if (progress.frames_total > 0)
        parts.push(`${Math.round((100 * progress.frames_done) / progress.frames_total)}%`);
    if (progress.fps !== null) parts.push(`${Math.round(progress.fps)} fps`);
    if (progress.pass_number > 1) parts.push('second pass with backup LED locations');
    return parts.join(' · ');
};

/** "1.2 GB" / "35.4 MB" / "800 KB". */
export const formatBytes = (bytes: number): string => {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
};

/** Start and stop video tests; every change refreshes the list. */
export const useVideoTestActions = () => {
    const client = useQueryClient();
    const refresh = () => {
        client.invalidateQueries({ queryKey: RUNS_KEY });
        client.invalidateQueries({ queryKey: LOG_KEY });
    };
    const onError = (fallback: string) => (error: unknown) =>
        toast.error(apiErrorMessage(error, fallback));

    const startMutation = createMutation(() => ({
        ...startVideoTestMutation(),
        onSuccess: (run: VideoTestRunView) => {
            refresh();
            toast.success(
                `Processing ${run.videos.length} video${run.videos.length === 1 ? '' : 's'} with "${run.model}"`
            );
        },
        onError: onError('Starting the video test failed')
    }));
    const stopMutation = createMutation(() => ({
        ...stopVideoTestMutation(),
        onSuccess: () => {
            refresh();
            toast.success('Video test stopped');
        },
        onError: onError('Stopping failed')
    }));

    return {
        startMutation,
        stopMutation,
        start: (files: File[], model: string) => startMutation.mutate({ body: { files, model } }),
        stop: (name: string) => stopMutation.mutate({ path: { name } })
    };
};
