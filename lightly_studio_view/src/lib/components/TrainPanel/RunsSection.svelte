<script lang="ts">
    import { untrack } from 'svelte';
    import type { TrainingRunView } from '$lib/api/lightly_studio_local';
    import { Spinner, Typography } from '$lib/components';
    import ConfirmDeleteDialog from '$lib/components/ConfirmDeleteDialog/ConfirmDeleteDialog.svelte';
    import {
        isActiveRun,
        useTrainingActions,
        useTrainingRuns
    } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';
    import RunLogDialog from './RunLogDialog.svelte';
    import RunRow from './RunRow.svelte';
    import TestModelDialog from './TestModelDialog.svelte';

    const query = useTrainingRuns();
    const { stop, remove, deleteMutation, startTest, testMutation } = useTrainingActions();
    const runs = $derived(query.data ?? []);

    let logOf = $state<string | null>(null);
    let toDelete = $state<string | null>(null);
    let testOf = $state<{ run: TrainingRunView; lead: string } | null>(null);
    const logRun = $derived(runs.find((r) => r.name === logOf));

    // A training that finishes while the panel is open gets the offer to test its model.
    let knownFinished: Set<string> | null = null;
    const exportedModel = (r: TrainingRunView) =>
        r.kind === 'train' && r.status === 'finished' && !!r.onnx_path;
    $effect(() => {
        const finished = runs.filter(exportedModel).map((r) => r.name);
        untrack(() => {
            if (knownFinished === null) {
                if (query.isSuccess) knownFinished = new Set(finished);
                return;
            }
            const fresh = runs.find(
                (r) => finished.includes(r.name) && !knownFinished!.has(r.name)
            );
            knownFinished = new Set(finished);
            if (fresh && !testOf) testOf = { run: fresh, lead: `Model ${fresh.name} is ready.` };
        });
    });
    $effect(() => {
        if (!deleteMutation.isPending && deleteMutation.isSuccess) toDelete = null;
    });
    $effect(() => {
        if (!testMutation.isPending && testMutation.isSuccess) testOf = null;
    });
</script>

<Typography variant="subtitle2" component="h3" className="mb-2 mt-4"
    >Training and test runs</Typography
>
{#if query.isLoading}
    <Spinner size="small" align="center" />
{:else if query.error}
    <Typography variant="body2" className="text-red-500"
        >Error loading runs: {query.error.message}</Typography
    >
{:else if runs.length === 0}
    <Typography variant="body2" className="text-muted-foreground">
        No runs yet. "Train model" starts one on a prepared dataset.
    </Typography>
{:else}
    <ul class="flex flex-col gap-2" data-testid="runs-list">
        {#each runs as run (run.name)}
            <RunRow
                {run}
                onLog={() => (logOf = run.name)}
                onStop={() => stop(run.name)}
                onDelete={() => (toDelete = run.name)}
                onTest={() => (testOf = { run, lead: '' })}
            />
        {/each}
    </ul>
{/if}

{#if logOf && logRun}
    <RunLogDialog name={logOf} active={isActiveRun(logRun)} onClose={() => (logOf = null)} />
{/if}
{#if testOf}
    <TestModelDialog
        model={testOf.run.name}
        classes={testOf.run.classes}
        lead={testOf.lead}
        isPending={testMutation.isPending}
        onConfirm={startTest}
        onCancel={() => (testOf = null)}
    />
{/if}
{#if toDelete}
    <ConfirmDeleteDialog
        title="Delete run?"
        description={`The run folder of "${toDelete}" with its checkpoints and log is removed. An exported model stays in the models folder.`}
        isPending={deleteMutation.isPending}
        onConfirm={() => toDelete && remove(toDelete)}
        onCancel={() => (toDelete = null)}
    />
{/if}
