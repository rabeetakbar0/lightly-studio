<script lang="ts">
    import type { PipelineModelView } from '$lib/api/lightly_studio_local';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Switch } from '$lib/components/ui/switch';
    import { Label } from '$lib/components/ui/label';
    import { compareRows } from './compareRows';

    interface Props {
        models: PipelineModelView[];
        onClose: () => void;
    }
    let { models, onClose }: Props = $props();

    let onlyDifferences = $state(false);
    const rows = $derived(compareRows(models));
    const shown = $derived(onlyDifferences ? rows.filter((r) => r.differs) : rows);
</script>

<Dialog.Root bind:open={() => true, onClose}>
    <Dialog.Content class="max-h-[90vh] max-w-5xl overflow-y-auto">
        <Dialog.Header>
            <Dialog.Title>Compare models</Dialog.Title>
            <Dialog.Description>
                Dataset, results and every training setting side by side. Rows where the models
                differ are highlighted; models trained outside the pipeline have no history.
            </Dialog.Description>
        </Dialog.Header>
        <div class="flex items-center gap-2 py-2">
            <Switch id="only-differences" bind:checked={onlyDifferences} />
            <Label for="only-differences">Only show differences</Label>
        </div>
        <table class="w-full text-sm" data-testid="compare-table">
            <thead>
                <tr class="border-b border-border text-left">
                    <th class="py-1 pr-3 font-medium">Setting</th>
                    {#each models as model (model.name)}
                        <th class="py-1 pr-3 font-medium">{model.name}</th>
                    {/each}
                </tr>
            </thead>
            <tbody>
                {#each shown as row (row.group + row.label)}
                    <tr
                        class={row.differs
                            ? 'border-b border-border bg-amber-500/10'
                            : 'border-b border-border'}
                        data-differs={row.differs}
                    >
                        <td class="py-1 pr-3 text-muted-foreground">
                            <span class="text-xs">{`${row.group} · `}</span>{row.label}
                        </td>
                        {#each row.values as value, i (i)}
                            <td class="py-1 pr-3 font-mono text-xs">{value}</td>
                        {/each}
                    </tr>
                {/each}
            </tbody>
        </table>
    </Dialog.Content>
</Dialog.Root>
