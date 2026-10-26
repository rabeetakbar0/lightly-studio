<script lang="ts">
    import type { CollectionView, CollectionViewWithCount } from '$lib/api/lightly_studio_local';
    import ConfirmDeleteDialog from '$lib/components/ConfirmDeleteDialog/ConfirmDeleteDialog.svelte';
    import { useDeleteAllImagesDialog } from '$lib/hooks/useDeleteAllImagesDialog/useDeleteAllImagesDialog';
    import { useDeleteImages } from '$lib/hooks/useDeleteImages/useDeleteImages.svelte';

    interface Props {
        /** The layout passes the details view, which carries the sample count. */
        collection: CollectionView | CollectionViewWithCount;
    }
    let { collection }: Props = $props();

    const { closeDeleteAllImagesDialog } = useDeleteAllImagesDialog();
    const { mutation, deleteAll } = useDeleteImages(() => ({
        collectionId: collection.collection_id,
        onDeleted: closeDeleteAllImagesDialog
    }));
    const count = $derived('total_sample_count' in collection ? collection.total_sample_count : 0);
</script>

<ConfirmDeleteDialog
    title="Delete all images?"
    description={`All ${count} images in "${collection.name}" are removed with their annotations, tags and uploaded files. You get an empty collection to drop new data into.`}
    confirmLabel="Delete everything"
    isPending={mutation.isPending}
    onConfirm={deleteAll}
    onCancel={closeDeleteAllImagesDialog}
/>
