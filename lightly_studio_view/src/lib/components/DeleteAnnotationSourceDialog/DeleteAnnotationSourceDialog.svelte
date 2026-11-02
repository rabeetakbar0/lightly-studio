<script lang="ts">
    import { untrack } from 'svelte';
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Checkbox } from '$lib/components/ui/checkbox';
    import { Label } from '$lib/components/ui/label';
    import { Select } from '$lib/components/Select';

    interface Props {
        sources: { id: string; name: string }[];
        isPending: boolean;
        onConfirm: (sourceId: string, deleteUnusedClasses: boolean) => void;
        onCancel: () => void;
    }
    let { sources, isPending, onConfirm, onCancel }: Props = $props();

    let sourceId = $state<string | undefined>(
        untrack(() => (sources.length === 1 ? sources[0].id : undefined))
    );
    let deleteUnusedClasses = $state(true);
    const name = $derived(sources.find((s) => s.id === sourceId)?.name ?? '');

    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content>
        <Dialog.Header>
            <Dialog.Title>Delete an annotation source?</Dialog.Title>
            <Dialog.Description>
                Removes every annotation in the source, the evaluations that compared against it,
                and the source itself. The images stay. This cannot be undone.
            </Dialog.Description>
        </Dialog.Header>
        <div class="grid gap-3 py-4">
            <div class="grid gap-1.5">
                <Label>Annotation source</Label>
                <Select
                    items={sources.map((s) => ({ value: s.id, label: s.name }))}
                    value={sourceId}
                    placeholder="Select a source"
                    onValueChange={(v) => (sourceId = v)}
                    class="w-full"
                    testId="delete-source-select"
                />
            </div>
            <div class="flex items-start gap-2">
                <Checkbox
                    id="delete-unused-classes"
                    bind:checked={deleteUnusedClasses}
                    data-testid="delete-unused-classes"
                />
                <Label for="delete-unused-classes" class="font-normal leading-snug">
                    Also delete the annotation classes that no other source uses any more
                </Label>
            </div>
        </div>
        <Dialog.Footer>
            <Button
                variant="outline"
                buttonProps={{ type: 'button', disabled: isPending, onclick: close }}>Cancel</Button
            >
            <Button
                variant="destructive"
                {isPending}
                buttonProps={{
                    type: 'button',
                    disabled: !sourceId || isPending,
                    onclick: () => sourceId && onConfirm(sourceId, deleteUnusedClasses),
                    'data-testid': 'delete-source-confirm'
                }}
            >
                {isPending ? 'Deleting…' : name ? `Delete "${name}"` : 'Delete'}
            </Button>
        </Dialog.Footer>
    </Dialog.Content>
</Dialog.Root>
