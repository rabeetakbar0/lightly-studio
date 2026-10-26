import { createMutation, useQueryClient } from '@tanstack/svelte-query';
import { invalidateAll } from '$app/navigation';
import { toast } from 'svelte-sonner';
import type { DeleteImagesView } from '$lib/api/lightly_studio_local';
import { deleteImagesMutation } from '$lib/api/lightly_studio_local/@tanstack/svelte-query.gen';
import { reloadDimensionBounds } from '$lib/hooks/useDimensions/useDimensions';
import { useTags } from '$lib/hooks/useTags/useTags';
import { apiErrorMessage } from '$lib/hooks/useMediaUpload/useMediaUpload.helpers';

interface UseDeleteImagesParams {
    collectionId: string;
    /** Runs after a successful delete, for example to close a dialog. */
    onDeleted?: (result: DeleteImagesView) => void;
}

/** Deletes images of a collection by tag, by id, or all of them, then refreshes every view. */
export const useDeleteImages = (getParams: () => UseDeleteImagesParams) => {
    const client = useQueryClient();

    const refresh = async (tagId?: string) => {
        const { collectionId } = getParams();
        // A delete touches counts, grids, labels and bounds alike; drop every cached query.
        await client.invalidateQueries();
        void invalidateAll();
        void reloadDimensionBounds(collectionId);
        const { loadTags, clearTagSelected } = useTags({
            collection_id: collectionId,
            autoLoad: false
        });
        if (tagId) clearTagSelected(tagId);
        await loadTags();
    };

    const mutation = createMutation(() => ({
        ...deleteImagesMutation(),
        onSuccess: async (result: DeleteImagesView, variables) => {
            await refresh(variables.body?.tag_id ?? undefined);
            const noun = result.deleted_samples === 1 ? 'image' : 'images';
            toast.success(`Deleted ${result.deleted_samples} ${noun}`);
            getParams().onDeleted?.(result);
        },
        onError: (error) => {
            toast.error(apiErrorMessage(error, 'Failed to delete images'));
        }
    }));

    const run = (body: { tag_id?: string; sample_ids?: string[]; delete_all?: boolean }) =>
        mutation.mutate({ path: { collection_id: getParams().collectionId }, body });

    return {
        mutation,
        /** Deletes every image carrying the tag, then the tag itself. */
        deleteByTag: (tagId: string) => run({ tag_id: tagId }),
        /** Deletes every image of the collection and its uploaded files. */
        deleteAll: () => run({ delete_all: true }),
        deleteSamples: (sampleIds: string[]) => run({ sample_ids: sampleIds })
    };
};
