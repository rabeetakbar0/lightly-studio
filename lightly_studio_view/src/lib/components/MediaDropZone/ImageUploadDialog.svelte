<script lang="ts">
    import { Button } from '$lib/components';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';

    interface Props {
        images: File[];
        onConfirm: (tag: string) => void;
        onCancel: () => void;
    }
    let { images, onConfirm, onCancel }: Props = $props();

    let tag = $state('');

    const submit = (event: SubmitEvent) => {
        event.preventDefault();
        onConfirm(tag.trim());
    };
</script>

<Dialog.Root bind:open={() => true, onCancel}>
    <Dialog.Content>
        <form onsubmit={submit}>
            <Dialog.Header>
                <Dialog.Title
                    >Add {images.length} {images.length === 1 ? 'image' : 'images'}</Dialog.Title
                >
                <Dialog.Description>
                    A tag groups these images in the filter panel, so they can be found or removed
                    together later. Leave it empty to add them without a tag.
                </Dialog.Description>
            </Dialog.Header>
            <div class="grid gap-2 py-4">
                <Label for="image-upload-tag">Tag (optional)</Label>
                <Input
                    id="image-upload-tag"
                    bind:value={tag}
                    placeholder="e.g. site-A-2026-10"
                    data-testid="image-tag-input"
                />
            </div>
            <Dialog.Footer>
                <Button variant="outline" buttonProps={{ type: 'button', onclick: onCancel }}>
                    Cancel
                </Button>
                <Button buttonProps={{ type: 'submit' }}>Add images</Button>
            </Dialog.Footer>
        </form>
    </Dialog.Content>
</Dialog.Root>
