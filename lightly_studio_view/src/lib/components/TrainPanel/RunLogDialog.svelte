<script lang="ts">
    import * as Dialog from '$lib/components/ui/dialog';
    import { Spinner } from '$lib/components';
    import { useTrainingLog } from '$lib/hooks/usePipelineTraining/usePipelineTraining.svelte';

    interface Props {
        name: string;
        active: boolean;
        onClose: () => void;
    }
    let { name, active, onClose }: Props = $props();

    const query = useTrainingLog(() => ({ name, active }));
    let pre = $state<HTMLPreElement | null>(null);

    // Keep the newest lines in view while the run writes.
    $effect(() => {
        if (query.data && pre) pre.scrollTop = pre.scrollHeight;
    });
</script>

<Dialog.Root bind:open={() => true, onClose}>
    <Dialog.Content class="max-w-4xl">
        <Dialog.Header>
            <Dialog.Title>Log of {name}</Dialog.Title>
            <Dialog.Description>
                The last 400 lines of log.txt in the run folder{active
                    ? ', refreshed every few seconds'
                    : ''}.
            </Dialog.Description>
        </Dialog.Header>
        {#if query.isLoading}
            <Spinner size="medium" align="center" />
        {:else if query.error}
            <p class="text-sm text-red-500">Cannot read the log: {query.error.message}</p>
        {:else}
            <pre
                bind:this={pre}
                class="max-h-[60vh] overflow-auto rounded-md bg-muted p-3 text-xs dark:[color-scheme:dark]"
                data-testid="run-log">{query.data?.text || 'The log is still empty.'}</pre>
        {/if}
    </Dialog.Content>
</Dialog.Root>
