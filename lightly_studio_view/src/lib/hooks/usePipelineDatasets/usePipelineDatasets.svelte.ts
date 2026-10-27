import { createMutation, createQuery, useQueryClient } from '@tanstack/svelte-query';
import { toast } from 'svelte-sonner';
import type { PipelineDatasetView } from '$lib/api/lightly_studio_local';
import {
    deletePipelineDatasetMutation,
    exportPipelineDatasetMutation,
    listPipelineDatasetsOptions,
    mergePipelineDatasetsMutation,
    previewPipelineDatasetMutation
} from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { apiErrorMessage } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

const LIST_KEY = [{ _id: 'listPipelineDatasets' }];

/** The prepared training datasets, newest first. */
export const usePipelineDatasets = () => createQuery(() => listPipelineDatasetsOptions());

/** Formats a byte count for the dataset list. */
export const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
};

/** Mutations over the datasets folder; every success refreshes the list. */
export const usePipelineDatasetActions = () => {
    const client = useQueryClient();
    const refresh = () => client.invalidateQueries({ queryKey: LIST_KEY });
    const onError = (fallback: string) => (error: unknown) =>
        toast.error(apiErrorMessage(error, fallback));

    const exportMutation = createMutation(() => ({
        ...exportPipelineDatasetMutation(),
        onSuccess: (result: PipelineDatasetView) => {
            refresh();
            toast.success(`Dataset "${result.name}" prepared with ${result.images} images`);
        },
        onError: onError('Preparing the dataset failed')
    }));
    const mergeMutation = createMutation(() => ({
        ...mergePipelineDatasetsMutation(),
        onSuccess: (result: PipelineDatasetView) => {
            refresh();
            toast.success(`Dataset "${result.name}" merged with ${result.images} images`);
        },
        onError: onError('Merging failed')
    }));
    const previewMutation = createMutation(() => ({
        ...previewPipelineDatasetMutation(),
        onError: onError('Preview failed')
    }));
    const deleteMutation = createMutation(() => ({
        ...deletePipelineDatasetMutation(),
        onSuccess: () => {
            refresh();
            toast.success('Dataset deleted');
        },
        onError: onError('Deleting failed')
    }));

    /** Loads the dataset into Studio and shows it in a new browser tab. */
    const preview = (name: string) => {
        // The tab is opened before the request, so popup blockers see a user gesture.
        const tab = window.open('', '_blank');
        previewMutation.mutate(
            { path: { name } },
            {
                onSuccess: (result) => {
                    const url = `/datasets/${result.collection_id}`;
                    if (tab) tab.location.href = url;
                    else window.open(url, '_blank');
                },
                onError: () => tab?.close()
            }
        );
    };

    return {
        exportMutation,
        mergeMutation,
        previewMutation,
        deleteMutation,
        preview,
        merge: (name: string, sources: string[]) =>
            mergeMutation.mutate({ body: { name, sources } }),
        remove: (name: string) => deleteMutation.mutate({ path: { name } })
    };
};
