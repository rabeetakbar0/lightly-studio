<script lang="ts">
    import { untrack } from 'svelte';
    import type { TrainConfig, TrainingOptionsView } from '$lib/api/lightly_studio_local';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import * as Tabs from '$lib/components/ui/tabs';
    import ConfigFieldGrid from './ConfigFieldGrid.svelte';
    import TrainRunFields from './TrainRunFields.svelte';
    import { RUN_NAME_PATTERN, TRAIN_FIELD_GROUPS } from './trainFields';

    interface Props {
        options: TrainingOptionsView;
        datasetNames: string[];
        isPending: boolean;
        /** Settings to start from instead of the defaults, e.g. those of an earlier model. */
        initialConfig?: Partial<TrainConfig>;
        onConfirm: (body: { name: string; dataset: string; config: TrainConfig }) => void;
        onCancel: () => void;
    }
    let { options, datasetNames, isPending, initialConfig, onConfirm, onCancel }: Props = $props();

    let config = $state<TrainConfig>(
        untrack(() => ({ ...options.defaults, ...(initialConfig ?? {}) }))
    );
    let name = $state('');
    let dataset = $state<string | undefined>(untrack(() => datasetNames[0]));

    const canSubmit = $derived(
        options.environment.ready && !!dataset && RUN_NAME_PATTERN.test(name.trim()) && !isPending
    );
    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (canSubmit && dataset) onConfirm({ name: name.trim(), dataset, config });
    };
    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content class="max-h-[90vh] max-w-3xl overflow-y-auto">
        <form onsubmit={submit} novalidate>
            <Dialog.Header>
                <Dialog.Title>Train a YOLOX model</Dialog.Title>
                <Dialog.Description>
                    Every setting starts at the v4 gate-arm recipe. The exp file and the log are
                    kept in the run folder; the exported model appears as a Detect operator.
                </Dialog.Description>
            </Dialog.Header>
            {#if !options.environment.ready}
                <div
                    class="mt-3 rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm"
                    data-testid="train-problems"
                >
                    Training is not configured on this machine:
                    <ul class="ml-4 list-disc">
                        {#each options.environment.problems as problem (problem)}
                            <li>{problem}</li>
                        {/each}
                    </ul>
                </div>
            {/if}
            <Tabs.Root value="run" class="py-4">
                <Tabs.List class="flex flex-wrap">
                    {#each TRAIN_FIELD_GROUPS as group (group.id)}
                        <Tabs.Trigger value={group.id} data-testid={`train-tab-${group.id}`}>
                            {group.label}
                        </Tabs.Trigger>
                    {/each}
                </Tabs.List>
                {#each TRAIN_FIELD_GROUPS as group (group.id)}
                    <Tabs.Content value={group.id} class="pt-3">
                        {#if group.id === 'run'}
                            <TrainRunFields
                                {options}
                                {datasetNames}
                                bind:name
                                bind:dataset
                                bind:config
                            />
                        {/if}
                        {#if group.id === 'model'}
                            <TrainRunFields {options} {datasetNames} bind:config presetOnly />
                        {/if}
                        <ConfigFieldGrid fields={group.fields} bind:config />
                    </Tabs.Content>
                {/each}
            </Tabs.Root>
            <Dialog.Footer>
                <Button
                    variant="outline"
                    buttonProps={{ type: 'button', disabled: isPending, onclick: close }}
                    >Cancel</Button
                >
                <Button
                    {isPending}
                    buttonProps={{
                        type: 'submit',
                        disabled: !canSubmit,
                        'data-testid': 'train-confirm'
                    }}
                >
                    {isPending ? 'Starting…' : 'Start training'}
                </Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
