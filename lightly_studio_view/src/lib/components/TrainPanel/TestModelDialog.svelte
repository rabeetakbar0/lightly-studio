<script lang="ts">
    import type { StartTestRequest } from '$lib/api/lightly_studio_local';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { Switch } from '$lib/components/ui/switch';
    import { Select } from '$lib/components/Select';
    import { useTestDatasets } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';

    interface Props {
        model: string;
        classes: string[];
        /** Shown above the form, e.g. "Model x is ready." */
        lead?: string;
        isPending: boolean;
        onConfirm: (body: StartTestRequest) => void;
        onCancel: () => void;
    }
    let { model, classes, lead = '', isPending, onConfirm, onCancel }: Props = $props();

    const query = useTestDatasets();
    let dataset = $state<string | undefined>(undefined);
    let batchSize = $state(16);
    let fp16 = $state(true);

    const sameClasses = (other: string[]) =>
        other.length === classes.length && other.every((c, i) => c === classes[i]);
    const items = $derived(
        (query.data ?? []).map((d) => ({
            value: d.name,
            label: sameClasses(d.classes)
                ? `${d.name} (${d.images} test images${d.kind === 'test' ? ', test set' : ''})`
                : `${d.name} (classes differ: ${d.classes.join(', ')})`,
            disabled: !sameClasses(d.classes)
        }))
    );
    // The first dataset with matching classes is preselected.
    $effect(() => {
        const first = items.find((i) => !i.disabled);
        if (dataset === undefined && first) dataset = first.value;
    });
    const canSubmit = $derived(!!dataset && batchSize >= 1 && !isPending);
    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (canSubmit && dataset) onConfirm({ model, dataset, batch_size: batchSize, fp16 });
    };
    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content>
        <form onsubmit={submit} novalidate>
            <Dialog.Header>
                <Dialog.Title>Test {model}</Dialog.Title>
                <Dialog.Description>
                    {lead}
                    Runs the YOLOX evaluator on the model's checkpoint over the test split of a dataset.
                    The scores go into the model's history, with a note when test images were part of
                    its training data.
                </Dialog.Description>
            </Dialog.Header>
            <div class="grid gap-3 py-4">
                <div class="grid gap-1.5">
                    <Label>Test dataset</Label>
                    {#if query.isSuccess && items.length === 0}
                        <p class="text-sm text-muted-foreground" data-testid="no-test-datasets">
                            No dataset has a test split yet. Prepare one with a test fraction, or
                            build a test dataset in the Train panel.
                        </p>
                    {:else}
                        <Select
                            {items}
                            value={dataset}
                            placeholder={query.isLoading ? 'Loading…' : 'Select a dataset'}
                            onValueChange={(v) => (dataset = v)}
                            class="w-full"
                            testId="test-dataset"
                        />
                    {/if}
                </div>
                <div class="grid gap-3 sm:grid-cols-2">
                    <div class="grid gap-1.5">
                        <Label for="test-batch">Batch size</Label>
                        <Input
                            id="test-batch"
                            type="number"
                            min={1}
                            step={1}
                            bind:value={batchSize}
                        />
                    </div>
                    <div class="flex items-center gap-2 pt-6">
                        <Switch id="test-fp16" bind:checked={fp16} />
                        <Label for="test-fp16">Mixed precision</Label>
                    </div>
                </div>
            </div>
            <Dialog.Footer>
                <Button
                    variant="outline"
                    buttonProps={{ type: 'button', disabled: isPending, onclick: close }}
                    >Not now</Button
                >
                <Button
                    {isPending}
                    buttonProps={{
                        type: 'submit',
                        disabled: !canSubmit,
                        'data-testid': 'test-confirm'
                    }}
                >
                    {isPending ? 'Starting…' : 'Run test'}
                </Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
