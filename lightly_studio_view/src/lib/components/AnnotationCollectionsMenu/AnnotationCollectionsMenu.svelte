<script lang="ts">
    import { Trash2 } from '@lucide/svelte';
    import Segment from '$lib/components/Segment/Segment.svelte';
    import { SideMenu } from '$lib/components/SideMenu';
    import DeleteAnnotationSourceDialog from '$lib/components/DeleteAnnotationSourceDialog/DeleteAnnotationSourceDialog.svelte';
    import { useAnnotationCollections } from '$lib/hooks/useAnnotationCollections/useAnnotationCollections';
    import { useAnnotationCollectionsFilter } from '$lib/hooks/useAnnotationCollectionsFilter/useAnnotationCollectionsFilter';
    import { useDeleteAnnotationSource } from '$lib/hooks/useDeleteAnnotationSource/useDeleteAnnotationSource.svelte';
    import { useSettings } from '$lib/hooks/useSettings';
    import { resolveEffectiveColorBySource } from '$lib/utils';

    interface Props {
        collectionId: string;
    }

    let { collectionId }: Props = $props();

    const annotationCollectionsQuery = useAnnotationCollections(() => ({ collectionId }));
    const items = $derived(
        (annotationCollectionsQuery.data ?? []).map((c) => ({ id: c.collection_id, name: c.name }))
    );

    const { setSelectedCollectionIds, selectedCollectionIds, multipleSourcesVisible } =
        useAnnotationCollectionsFilter();
    const { enforceColoringByClassStore } = useSettings();

    let deleteOpen = $state(false);
    const { mutation, deleteSource } = useDeleteAnnotationSource(() => ({
        collectionId,
        onDeleted: () => (deleteOpen = false)
    }));

    // Checkboxes are only worth showing when there is a choice to make. The selection itself is
    // filled by useSeedAnnotationSourceFilter in the collection layout, which runs for every
    // collection including those with a single source. A single source is still listed, so
    // that it can be deleted.
    const hasChoice = $derived(items.length > 1);
</script>

{#if items.length > 0}
    <Segment title="Annotation Sources">
        {#if hasChoice}
            <SideMenu
                showColorMarker={resolveEffectiveColorBySource({
                    multipleSourcesVisible: $multipleSourcesVisible,
                    enforceColoringByClass: $enforceColoringByClassStore
                })}
                enableColorPicker
                {items}
                selectedItemsIds={$selectedCollectionIds}
                onChangeSelectedItems={setSelectedCollectionIds}
            />
        {:else}
            <p class="text-sm text-foreground" data-testid="single-annotation-source">
                {items[0].name}
            </p>
        {/if}
        <button
            type="button"
            class="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground hover:text-destructive"
            onclick={() => (deleteOpen = true)}
            data-testid="delete-annotation-source"
        >
            <Trash2 class="size-3.5" />
            Delete an annotation source…
        </button>
    </Segment>
{/if}

{#if deleteOpen}
    <DeleteAnnotationSourceDialog
        sources={items}
        isPending={mutation.isPending}
        onConfirm={deleteSource}
        onCancel={() => (deleteOpen = false)}
    />
{/if}
