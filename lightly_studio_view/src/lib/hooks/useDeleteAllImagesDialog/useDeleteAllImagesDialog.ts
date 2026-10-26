import { writable } from 'svelte/store';

const isDeleteAllImagesDialogOpen = writable(false);

/** Opens and closes the header menu's "Delete all images" confirmation. */
export function useDeleteAllImagesDialog() {
    return {
        isDeleteAllImagesDialogOpen,
        openDeleteAllImagesDialog: () => isDeleteAllImagesDialogOpen.set(true),
        closeDeleteAllImagesDialog: () => isDeleteAllImagesDialogOpen.set(false)
    };
}
