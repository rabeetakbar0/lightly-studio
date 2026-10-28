import { type AnnotationUpdateInput } from '$lib/api/lightly_studio_local';
import { updateAnnotationsMutation } from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { useImageAnnotationCountsQueryKey } from '$lib/hooks/useImageAnnotationCounts/useImageAnnotationCounts';
import { useInvalidateAnnotationGridQueries } from '$lib/hooks/useInvalidateAnnotationGridQueries';
import { useInvalidateEvaluationRunsQueries } from '$lib/hooks/useEvaluationRuns/useEvaluationRuns';

export const useUpdateAnnotationsMutation = ({
    getCollectionId
}: {
    getCollectionId: () => string;
}) => {
    const mutation = createMutation(() => updateAnnotationsMutation());

    const client = useQueryClient();
    const invalidateAnnotationGridQueries = useInvalidateAnnotationGridQueries();
    const invalidateEvaluationRunsQueries = useInvalidateEvaluationRunsQueries();

    const refetch = (collectionId: string) => {
        invalidateAnnotationGridQueries(collectionId);
        client.invalidateQueries({
            queryKey: useImageAnnotationCountsQueryKey
        });
        // Annotation mutations can mark evaluation runs as stale, so refresh the runs list.
        invalidateEvaluationRunsQueries();
    };

    const updateAnnotations = async (inputs: AnnotationUpdateInput[]) => {
        const collectionId = getCollectionId();
        await mutation.mutateAsync({
            path: {
                collection_id: collectionId
            },
            body: inputs
        });
        refetch(collectionId);
    };

    return {
        updateAnnotations
    };
};
