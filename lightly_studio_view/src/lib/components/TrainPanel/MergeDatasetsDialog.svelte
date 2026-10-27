<script lang="ts">
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import { Select } from '$lib/components/Select';

    interface Props {
        datasetNames: string[];
        isPending: boolean;
        onConfirm: (name: string, sources: string[]) => void;
        onCancel: () => void;
    }
    let { datasetNames, isPending, onConfirm, onCancel }: Props = $props();

    let name = $state('');
    let first = $state<string | undefined>(undefined);
    let second = $state<string | undefined>(undefined);

    const items = (excluded: string | undefined) =>
        datasetNames.map((n) => ({ value: n, label: n, disabled: n === excluded }));
    const canSubmit = $derived(
        !!first &&
            !!second &&
            first !== second &&
            /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name.trim()) &&
            !isPending
    );
    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        if (canSubmit && first && second) onConfirm(name.trim(), [first, second]);
    };
    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content>
        <form onsubmit={submit} novalidate>
            <Dialog.Header>
                <Dialog.Title>Merge two datasets</Dialog.Title>
                <Dialog.Description>
                    Each source keeps its own train and validation split. Classes are matched by
                    name. The sources stay unchanged.
                </Dialog.Description>
            </Dialog.Header>
            <div class="grid gap-3 py-4">
                <div class="grid gap-1.5">
                    <Label>First dataset</Label>
                    <Select
                        items={items(second)}
                        value={first}
                        placeholder="Select a dataset"
                        onValueChange={(v) => (first = v)}
                        class="w-full"
                        testId="merge-first"
                    />
                </div>
                <div class="grid gap-1.5">
                    <Label>Second dataset</Label>
                    <Select
                        items={items(first)}
                        value={second}
                        placeholder="Select a dataset"
                        onValueChange={(v) => (second = v)}
                        class="w-full"
                        testId="merge-second"
                    />
                </div>
                <div class="grid gap-1.5">
                    <Label for="merge-name">New dataset name</Label>
                    <Input
                        id="merge-name"
                        bind:value={name}
                        placeholder="e.g. data1_plus_data2"
                        data-testid="merge-name"
                    />
                </div>
            </div>
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
                        'data-testid': 'merge-confirm'
                    }}
                >
                    {isPending ? 'Merging…' : 'Merge'}
                </Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
