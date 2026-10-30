<script lang="ts">
    import { page } from '$app/state';
    import { Info } from '@lucide/svelte';
    import AddStrategyButton from '$lib/components/Sampling/AddStrategyButton.svelte';
    import StrategyCard from '$lib/components/Sampling/StrategyCard/StrategyCard.svelte';
    import FieldTooltip from '$lib/components/FieldTooltip/FieldTooltip.svelte';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { useSamplingDialog } from '$lib/hooks/useSamplingDialog/useSamplingDialog';
    import { useStrategyBuilder } from '$lib/hooks/useStrategyBuilder';
    import { useSamplingCombinationDialog } from './useSamplingCombinationDialog/useSamplingCombinationDialog';
    import { useStrategyOptions } from './useSamplingCombinationDialog/useStrategyOptions.svelte';
    import SampleCountInput from '$lib/components/Sampling/SampleCountInput/SampleCountInput.svelte';
    import { PreselectedTagField } from './PreselectedTagField';

    const collectionId = $derived(page.params.collection_id!);
    const isVideoCollection = $derived(
        page.data.collection?.sample_type === 'video' ||
            page.data.collection?.sample_type === 'video_frame'
    );
    const isSequenceCollection = $derived(page.data.collection?.sample_type === 'sequence');
    // Subpart diversity needs annotation crop embeddings, which are only generated for images.
    const lacksCropEmbeddings = $derived(isVideoCollection || isSequenceCollection);

    const { isSamplingDialogOpen, openSamplingDialog, closeSamplingDialog } = useSamplingDialog();

    const {
        instances,
        addStrategy,
        duplicateStrategy,
        removeStrategy,
        resetStrategies,
        toggleExpand,
        updateParams
    } = useStrategyBuilder();

    const strategyOptions = useStrategyOptions(() => collectionId);

    const {
        tags,
        nSamplesToSelect,
        percentageToSelect,
        updateAbsolute,
        updatePercentage,
        selectionResultTagName,
        preselectedTagId,
        filteredSampleCount,
        noSamples,
        notEnoughSamples,
        sampleCountLabel,
        createButtonTooltip,
        isSubmitting,
        loadingMessage,
        handleFormSubmit
    } = useSamplingCombinationDialog({
        getCollectionId: () => collectionId,
        getIsVideoCollection: () => isVideoCollection,
        getIsSequenceCollection: () => isSequenceCollection,
        instances,
        onSubmitSuccess: resetStrategies
    });

    // TODO(Leonardo, 06/2026): Update once there are multiple embedding models - currently only one diversity
    // strategy is supported since all samples share a single embedding space.
    const hasDiversity = $derived($instances.some((i) => i.type === 'diversity'));
    const hasDeduplication = $derived($instances.some((i) => i.type === 'deduplication'));
    const hasSubpartDiversity = $derived($instances.some((i) => i.type === 'subpart_diversity'));
</script>

<Dialog.Root
    open={$isSamplingDialogOpen}
    onOpenChange={(open) => (open ? openSamplingDialog() : closeSamplingDialog())}
>
    <Dialog.Portal>
        <Dialog.Overlay />
        <Dialog.Content class="border-border bg-background sm:max-w-[560px]">
            <form onsubmit={handleFormSubmit}>
                <Dialog.Header>
                    <Dialog.Title class="text-foreground">Create Sampling</Dialog.Title>
                    <Dialog.Description class="text-foreground">
                        Sample from the <strong class="font-semibold text-primary"
                            >{$sampleCountLabel}</strong
                        > currently matching your filters.
                    </Dialog.Description>
                </Dialog.Header>

                <div class="max-h-[60vh] overflow-y-auto p-2 dark:[color-scheme:dark]">
                    <div class="grid gap-4 py-4">
                        <div class="w-full">
                            <AddStrategyButton
                                diversityDisabledReason={hasDiversity
                                    ? 'Only one diversity strategy can be added per selection.'
                                    : undefined}
                                deduplicationDisabledReason={hasDeduplication
                                    ? 'Only one deduplication strategy can be added per selection.'
                                    : undefined}
                                similarityDisabledReason={isVideoCollection
                                    ? 'Not available for video collections. Similarity selection requires image embeddings.'
                                    : $tags.length === 0
                                      ? 'No sample tags in this collection. Create a sample tag first to use as the similarity reference.'
                                      : undefined}
                                metadataWeightingDisabledReason={!strategyOptions.hasMetadataFields
                                    ? 'No numeric metadata fields found. Index metadata on your samples to enable this strategy.'
                                    : undefined}
                                classBalancingDisabledReason={isSequenceCollection
                                    ? 'Not available for sequence collections.'
                                    : !strategyOptions.hasAnnotationLabels
                                      ? 'No annotation labels found. Add annotations to your samples to enable this strategy.'
                                      : undefined}
                                metadataBalancingDisabledReason={!strategyOptions.hasCategoricalMetadataFields
                                    ? 'No categorical metadata fields found. Index string or boolean metadata on your samples to enable this strategy.'
                                    : undefined}
                                subpartDiversityDisabledReason={lacksCropEmbeddings
                                    ? 'Not available for video and sequence collections. Subpart diversity requires annotation crop embeddings, which are only generated for images.'
                                    : !strategyOptions.hasCroppableAnnotationSources
                                      ? 'No object detection or segmentation annotation sources found. Add object detection or segmentation annotations to enable this strategy.'
                                      : hasSubpartDiversity
                                        ? 'Only one subpart diversity strategy can be added per selection.'
                                        : undefined}
                                onAdd={addStrategy}
                            />
                        </div>

                        {#if $instances.length > 0}
                            <div class="grid gap-3">
                                <Label class="text-foreground">Strategies</Label>
                                {#each $instances as instance (instance.id)}
                                    <StrategyCard
                                        {instance}
                                        tags={$tags}
                                        annotationLabels={strategyOptions.annotationLabels}
                                        annotationSourceOptions={strategyOptions.annotationSourceOptions}
                                        croppableAnnotationSourceOptions={strategyOptions.croppableAnnotationSourceOptions}
                                        metadataFieldNames={strategyOptions.metadataFieldNames}
                                        categoricalMetadataFieldNames={strategyOptions.categoricalMetadataFieldNames}
                                        metadataValuesByKey={strategyOptions.metadataValuesByKey}
                                        isDuplicateDisabled={instance.type === 'diversity' ||
                                            instance.type === 'deduplication' ||
                                            instance.type === 'subpart_diversity'}
                                        onRemove={() => removeStrategy(instance.id)}
                                        onDuplicate={() => duplicateStrategy(instance.id)}
                                        onUpdate={(params) => updateParams(instance.id, params)}
                                        onToggleExpand={() => toggleExpand(instance.id)}
                                    />
                                {/each}
                                <span class="flex items-center gap-1 text-xs text-muted-foreground">
                                    <Info class="size-3 shrink-0" />
                                    The order of strategies does not affect the result.
                                </span>
                            </div>
                        {/if}

                        <div class="grid gap-2">
                            <div class="flex items-center gap-1.5">
                                <Label for="n-samples" class="text-foreground"
                                    >Number of Samples</Label
                                >
                                <FieldTooltip
                                    content="How many samples will be written to the output tag. Cannot exceed the number of samples matching the current filters."
                                />
                            </div>
                            <SampleCountInput
                                count={$nSamplesToSelect}
                                percentage={$percentageToSelect}
                                onCountChange={updateAbsolute}
                                onPercentageChange={updatePercentage}
                            />
                        </div>

                        <div class="grid gap-2">
                            <div class="flex items-center gap-1.5">
                                <Label for="tag-name" class="text-foreground">Tag Name</Label>
                                <FieldTooltip
                                    content="A new sample tag will be created with this name to store the selection result."
                                />
                            </div>
                            <Input
                                id="tag-name"
                                type="text"
                                bind:value={$selectionResultTagName}
                                placeholder="Enter a tag for the sampled subset"
                                required
                                data-testid="sampling-dialog-tag-name-input"
                            />
                        </div>

                        <!-- TODO: Validate the preselected tag against the active filters before allowing submission. -->
                        <PreselectedTagField
                            tags={$tags}
                            value={$preselectedTagId}
                            onValueChange={(value) => preselectedTagId.set(value)}
                        />

                        {#if $noSamples}
                            <p
                                class="text-sm text-destructive-text"
                                data-testid="sampling-dialog-no-samples-warning"
                            >
                                No samples match the current filters.
                            </p>
                        {/if}

                        {#if $notEnoughSamples}
                            <p
                                class="text-sm text-destructive-text"
                                data-testid="sampling-dialog-not-enough-samples-warning"
                            >
                                Only {$filteredSampleCount} samples are available, but
                                {$nSamplesToSelect} were requested.
                            </p>
                        {/if}
                    </div>
                </div>

                <Dialog.Footer class="mt-4">
                    <Button
                        variant="outline"
                        buttonProps={{
                            type: 'button',
                            onclick: closeSamplingDialog,
                            disabled: $isSubmitting,
                            'data-testid': 'sampling-dialog-cancel'
                        }}
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="default"
                        isPending={$isSubmitting}
                        disabledReason={$createButtonTooltip}
                        buttonProps={{
                            type: 'submit',
                            disabled: $notEnoughSamples || $noSamples,
                            'data-testid': 'sampling-dialog-submit'
                        }}
                    >
                        {$isSubmitting ? $loadingMessage || 'Creating...' : 'Create Selection'}
                    </Button>
                </Dialog.Footer>
            </form>
        </Dialog.Content>
    </Dialog.Portal>
</Dialog.Root>
