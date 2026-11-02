<script lang="ts">
    import { page } from '$app/state';
    import { AnnotationSourceSelect, Button, FormField } from '$lib/components';
    import { Input } from '$lib/components/ui/input';
    import { Select } from '$lib/components/Select';
    import { useImageFilters } from '$lib/hooks';
    import {
        splitSummary,
        usePipelineDatasetActions
    } from '$lib/hooks/usePipelineDatasets/usePipelineDatasets.svelte';

    interface Props {
        annotationSources: { id: string; name: string }[];
        selectedAnnotationCollectionId: string | undefined;
    }
    let { annotationSources, selectedAnnotationCollectionId = $bindable() }: Props = $props();

    const collectionId = page.params.collection_id!;
    const { imageFilter } = useImageFilters();
    const { exportMutation } = usePipelineDatasetActions();

    let name = $state('');
    let valFraction = $state(0.1);
    let testFraction = $state(0.1);
    let splitBy = $state<'tag' | 'image'>('tag');
    let splitStrategy = $state<'embedding' | 'random'>('embedding');
    let groups = $state<number | null>(null);

    const nameError = $derived(
        name.trim() === '' || /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name.trim())
            ? ''
            : 'Use letters, digits, ".", "_" or "-", starting with a letter or digit.'
    );
    const fractionError = $derived.by(() => {
        const ok = (v: number) => Number.isFinite(v) && v >= 0 && v < 1;
        if (!ok(valFraction) || !ok(testFraction))
            return 'Fractions must be at least 0 and below 1.';
        if (valFraction + testFraction >= 1)
            return 'Validation and test together must stay below 1.';
        return '';
    });
    const holdsOut = $derived(valFraction > 0 || testFraction > 0);
    const sourceId = $derived(selectedAnnotationCollectionId ?? annotationSources[0]?.id);
    const canSubmit = $derived(
        name.trim() !== '' &&
            !nameError &&
            !fractionError &&
            !!sourceId &&
            !exportMutation.isPending
    );

    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (!canSubmit || !sourceId) return;
        exportMutation.mutate({
            path: { collection_id: collectionId },
            body: {
                name: name.trim(),
                annotation_collection_id: sourceId,
                image_filter: $imageFilter ?? null,
                val_fraction: valFraction,
                test_fraction: testFraction,
                split_by: splitBy,
                split_strategy: splitStrategy,
                groups: splitStrategy === 'embedding' && groups ? groups : null
            }
        });
    };
</script>

<form onsubmit={submit} class="flex flex-col gap-3" novalidate>
    <p class="text-sm text-muted-foreground">
        Writes the images in the current view and their boxes as a COCO dataset folder ready for
        training, with train, validation and test splits and bare file names. Images are hard
        linked, so nothing is duplicated or moved.
    </p>
    {#if annotationSources.length > 1}
        <FormField label="Annotation Source">
            <AnnotationSourceSelect
                sourceOptions={annotationSources}
                placeholder="Boxes from this source are exported"
                bind:selectedSource={selectedAnnotationCollectionId}
            />
        </FormField>
    {/if}
    <FormField label="Dataset name">
        <Input bind:value={name} placeholder="e.g. data1" data-testid="prepare-dataset-name" />
        {#if nameError}<p class="text-xs text-destructive">{nameError}</p>{/if}
    </FormField>
    <div class="grid grid-cols-2 gap-3">
        <FormField label="Validation fraction">
            <Input
                type="number"
                min={0}
                max={0.99}
                step={0.05}
                bind:value={valFraction}
                data-testid="prepare-val-fraction"
            />
        </FormField>
        <FormField label="Test fraction">
            <Input
                type="number"
                min={0}
                max={0.99}
                step={0.05}
                bind:value={testFraction}
                data-testid="prepare-test-fraction"
            />
        </FormField>
    </div>
    {#if fractionError}<p class="text-xs text-destructive">{fractionError}</p>{/if}
    <div class="grid grid-cols-2 gap-3">
        <FormField label="Hold-out spread">
            <Select
                items={[
                    { value: 'embedding', label: 'Spread over similarity groups' },
                    { value: 'random', label: 'Random' }
                ]}
                value={splitStrategy}
                disabled={!holdsOut}
                onValueChange={(v) => (splitStrategy = v as 'embedding' | 'random')}
                class="w-full"
                testId="prepare-split-strategy"
            />
        </FormField>
        <FormField label="Groups">
            <Input
                type="number"
                min={2}
                step={1}
                placeholder="auto"
                disabled={!holdsOut || splitStrategy !== 'embedding'}
                value={groups ?? ''}
                oninput={(e) =>
                    (groups = e.currentTarget.value ? Number(e.currentTarget.value) : null)}
                data-testid="prepare-groups"
            />
        </FormField>
    </div>
    <p class="text-xs text-muted-foreground">
        Images are grouped by visual similarity and both hold-out shares are taken from every group,
        so validation and test mirror the whole dataset. Missing embeddings are computed first.
        "Tag" keeps whole recordings in one split, so a hold-out can cover at most round(fraction ×
        recordings) of them.
    </p>
    <div class="grid grid-cols-2 gap-3">
        <FormField label="Split by">
            <Select
                items={[
                    { value: 'tag', label: 'Tag (whole recordings)' },
                    { value: 'image', label: 'Single images' }
                ]}
                value={splitBy}
                onValueChange={(v) => (splitBy = v as 'tag' | 'image')}
                class="w-full"
                testId="prepare-split-by"
            />
        </FormField>
    </div>
    {#if exportMutation.data}
        <p class="text-sm" data-testid="prepare-dataset-result">
            Prepared <strong>{exportMutation.data.name}</strong>: {exportMutation.data.images} images,
            {exportMutation.data.boxes} boxes, classes {exportMutation.data.classes.join(
                ', '
            )}{splitSummary(exportMutation.data.split)}. Open the Train panel to preview, merge or
            train on it.
        </p>
    {/if}
    <Button
        variant="default"
        isPending={exportMutation.isPending}
        buttonProps={{
            type: 'submit',
            class: 'my-2 w-full',
            disabled: !canSubmit,
            'data-testid': 'submit-button-prepare-dataset'
        }}
    >
        {exportMutation.isPending ? 'Preparing…' : 'Prepare dataset'}
    </Button>
</form>
