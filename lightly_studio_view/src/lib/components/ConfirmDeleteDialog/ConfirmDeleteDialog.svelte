<script lang="ts">
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';

    interface Props {
        title: string;
        description: string;
        confirmLabel?: string;
        isPending?: boolean;
        onConfirm: () => void;
        onCancel: () => void;
    }
    let {
        title,
        description,
        confirmLabel = 'Delete',
        isPending = false,
        onConfirm,
        onCancel
    }: Props = $props();

    // The dialog stays up while the delete runs, so the outcome is not missed.
    const close = () => {
        if (!isPending) onCancel();
    };
</script>

<Dialog.Root bind:open={() => true, close}>
    <Dialog.Content>
        <Dialog.Header>
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Description>{description} This cannot be undone.</Dialog.Description>
        </Dialog.Header>
        <Dialog.Footer>
            <Button
                variant="outline"
                buttonProps={{ type: 'button', disabled: isPending, onclick: close }}
            >
                Cancel
            </Button>
            <Button
                variant="destructive"
                {isPending}
                buttonProps={{
                    type: 'button',
                    disabled: isPending,
                    onclick: onConfirm,
                    'data-testid': 'confirm-delete'
                }}
            >
                {isPending ? 'Deleting…' : confirmLabel}
            </Button>
        </Dialog.Footer>
    </Dialog.Content>
</Dialog.Root>
