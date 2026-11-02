import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { invalidateAll } from '$app/navigation';
import { get } from 'svelte/store';
import { toast } from 'svelte-sonner';
import type { DeleteAnnotationSourceView } from '$lib/api/lightly_studio_local';
import { deleteAnnotationSourceMutation } from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { useAnnotationCollectionsFilter } from '$lib/hooks/useAnnotationCollectionsFilter/useAnnotationCollectionsFilter';
import { apiErrorMessage } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

interface UseDeleteAnnotationSourceParams {
    collectionId: string;
    /** Runs after a successful delete, for example to close a dialog. */
    onDeleted?: (result: DeleteAnnotationSourceView) => void;
}

/** "120 annotations, 1 evaluation and 2 unused classes" for the success toast. */
export const deletedSummary = (result: DeleteAnnotationSourceView): string => {
    const plural = (n: number, word: string, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;
    const parts = [plural(result.deleted_annotations, 'annotation')];
    if (result.deleted_evaluations) parts.push(plural(result.deleted_evaluations, 'evaluation'));
    if (result.deleted_classes)
        parts.push(plural(result.deleted_classes, 'unused class', 'unused classes'));
    return parts.length > 1
        ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
        : parts[0];
};

/** Deletes an annotation source with everything under it, then refreshes every view. */
export const useDeleteAnnotationSource = (getParams: () => UseDeleteAnnotationSourceParams) => {
    const client = useQueryClient();
    const { selectedCollectionIds, setSelectedCollectionIds } = useAnnotationCollectionsFilter();

    const mutation = createMutation(() => ({
        ...deleteAnnotationSourceMutation(),
        onSuccess: async (result: DeleteAnnotationSourceView, variables) => {
            const removed = variables.body.annotation_collection_id;
            setSelectedCollectionIds(get(selectedCollectionIds).filter((id) => id !== removed));
            // Sources, classes, counts, grids and evaluations all change; drop every cache.
            await client.invalidateQueries();
            void invalidateAll();
            toast.success(`Annotation source deleted with ${deletedSummary(result)}`);
            getParams().onDeleted?.(result);
        },
        onError: (error) => {
            toast.error(apiErrorMessage(error, 'Failed to delete the annotation source'));
        }
    }));

    return {
        mutation,
        deleteSource: (annotationCollectionId: string, deleteUnusedClasses: boolean) =>
            mutation.mutate({
                path: { collection_id: getParams().collectionId },
                body: {
                    annotation_collection_id: annotationCollectionId,
                    delete_unused_classes: deleteUnusedClasses
                }
            })
    };
};
