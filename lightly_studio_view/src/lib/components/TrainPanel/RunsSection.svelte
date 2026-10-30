<script lang="ts">
    import { Spinner, Typography } from '$lib/components';
    import ConfirmDeleteDialog from '$lib/components/ConfirmDeleteDialog/ConfirmDeleteDialog.svelte';
    import {
        useTrainingActions,
        useTrainingRuns
    } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';
    import { isActiveRun } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';
    import RunLogDialog from './RunLogDialog.svelte';
    import RunRow from './RunRow.svelte';

    const query = useTrainingRuns();
    const { stop, remove, deleteMutation } = useTrainingActions();
    const runs = $derived(query.data ?? []);

    let logOf = $state<string | null>(null);
    let toDelete = $state<string | null>(null);
    const logRun = $derived(runs.find((r) => r.name === logOf));

    $effect(() => {
        if (!deleteMutation.isPending && deleteMutation.isSuccess) toDelete = null;
    });
</script>

<Typography variant="subtitle2" component="h3" className="mb-2 mt-4">Training runs</Typography>
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
            />
        {/each}
    </ul>
{/if}

{#if logOf && logRun}
    <RunLogDialog name={logOf} active={isActiveRun(logRun)} onClose={() => (logOf = null)} />
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
