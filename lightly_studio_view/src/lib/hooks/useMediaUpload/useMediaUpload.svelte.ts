import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { invalidateAll } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { planVideoFrames, type MediaUploadView } from '$lib/api/lightly_studio_local';
import {
    uploadImagesMutation,
    uploadVideosMutation
} from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { getImagesInfiniteQueryKeyPrefix } from '$lib/hooks/useImagesInfinite/createImagesInfiniteOptions';
import { reloadDimensionBounds } from '$lib/hooks/useDimensions/useDimensions';
import { apiErrorMessage, type FrameSettings } from './useMediaUpload.helpers';

interface UseMediaUploadParams {
    collectionId: string;
}

const TOAST_ID = 'media-upload';

/** Uploads dropped images and videos into an image collection and refreshes the grid. */
export const useMediaUpload = (getParams: () => UseMediaUploadParams) => {
    const client = useQueryClient();

    const onSuccess = (result: MediaUploadView, noun: string) => {
        client.invalidateQueries({
            queryKey: getImagesInfiniteQueryKeyPrefix(getParams().collectionId)
        });
        void invalidateAll();
        void reloadDimensionBounds(getParams().collectionId);
        // Tags created by the upload show up in the filter panel.
        client.invalidateQueries({ queryKey: [{ _id: 'getTagsByCollectionId' }] });
        const skipped = result.skipped > 0 ? `, ${result.skipped} skipped` : '';
        toast.success(`Added ${result.added_sample_ids.length} ${noun}${skipped}`, {
            id: TOAST_ID
        });
        for (const warning of result.warnings) toast.warning(warning);
    };
    const onError = (error: unknown) => {
        toast.error(apiErrorMessage(error, 'Upload failed'), { id: TOAST_ID });
    };

    const imagesMutation = createMutation(() => ({
        ...uploadImagesMutation(),
        onSuccess: (result: MediaUploadView) => onSuccess(result, 'images'),
        onError
    }));
    const videosMutation = createMutation(() => ({
        ...uploadVideosMutation(),
        onSuccess: (result: MediaUploadView) => onSuccess(result, 'video frames'),
        onError
    }));

    /** Uploads images; a non-empty tag is put on all of them. */
    const uploadImages = (files: File[], tag = '') => {
        toast.loading(`Uploading ${files.length} images…`, { id: TOAST_ID });
        imagesMutation.mutate({
            path: { collection_id: getParams().collectionId },
            body: { files, tag }
        });
    };

    /** Uploads videos; tags has one entry per file, empty for no tag. */
    const uploadVideos = (files: File[], settings: FrameSettings, tags: string[]) => {
        toast.loading(`Extracting frames from ${files.length} videos…`, { id: TOAST_ID });
        videosMutation.mutate({
            path: { collection_id: getParams().collectionId },
            body: {
                files,
                tags,
                edge_window_seconds: settings.edgeWindowSeconds,
                edge_interval_seconds: settings.edgeIntervalSeconds,
                middle_frames: settings.middleFrames,
                max_frames: settings.maxFrames
            }
        });
    };

    /** Asks the backend how many frames the settings yield for videos of the given lengths. */
    const planFrames = async (durations: number[], settings: FrameSettings): Promise<number> => {
        const { data } = await planVideoFrames({
            body: {
                durations_seconds: durations,
                edge_window_seconds: settings.edgeWindowSeconds,
                edge_interval_seconds: settings.edgeIntervalSeconds,
                middle_frames: settings.middleFrames,
                max_frames: settings.maxFrames
            },
            throwOnError: true
        });
        return data.total_frames;
    };

    return {
        uploadImages,
        uploadVideos,
        planFrames,
        get isPending() {
            return imagesMutation.isPending || videosMutation.isPending;
        }
    };
};
