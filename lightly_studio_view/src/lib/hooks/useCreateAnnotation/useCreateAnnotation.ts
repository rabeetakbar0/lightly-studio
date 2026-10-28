import {
    type AnnotationCreateInput,
    type CreateAnnotationResponse
} from '$lib/api/lightly_studio_local';
import {
    createAnnotationMutation,
    readAnnotationCollectionsQueryKey
} from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { useImageAnnotationCountsQueryKey } from '$lib/hooks/useImageAnnotationCounts/useImageAnnotationCounts';
import { useInvalidateAnnotationGridQueries } from '$lib/hooks/useInvalidateAnnotationGridQueries';
import { useInvalidateEvaluationRunsQueries } from '$lib/hooks/useEvaluationRuns/useEvaluationRuns';

export const useCreateAnnotation = ({ getCollectionId }: { getCollectionId: () => string }) => {
    const mutation = createMutation(() => createAnnotationMutation());
    const client = useQueryClient();
    const invalidateAnnotationGridQueries = useInvalidateAnnotationGridQueries();
    const invalidateEvaluationRunsQueries = useInvalidateEvaluationRunsQueries();

    const refetch = (collectionId: string) => {
        invalidateAnnotationGridQueries(collectionId);
        client.invalidateQueries({
            queryKey: useImageAnnotationCountsQueryKey
        });
        // An annotation can be written to a brand-new source (collection) by name, so refresh
        // the source list to surface it in the pill and the side panel.
        client.invalidateQueries({
            queryKey: readAnnotationCollectionsQueryKey({ path: { collection_id: collectionId } })
        });
        // Annotation mutations can mark evaluation runs as stale, so refresh the runs list.
        invalidateEvaluationRunsQueries();
    };

    // Pass `skipRefetch` to create many annotations and call `refetch` once at the end.
    const createAnnotation = (
        inputs: AnnotationCreateInput,
        { skipRefetch = false }: { skipRefetch?: boolean } = {}
    ) =>
        new Promise<CreateAnnotationResponse>((resolve, reject) => {
            const collectionId = getCollectionId();
            mutation.mutate(
                {
                    path: {
                        collection_id: collectionId
                    },
                    body: inputs
                },
                {
                    onSuccess: (data) => {
                        if (!skipRefetch) refetch(collectionId);
                        resolve(data);
                    },
                    onError: (error) => {
                        reject(error);
                    }
                }
            );
        });

    return {
        createAnnotation,
        refetch: () => refetch(getCollectionId())
    };
};
