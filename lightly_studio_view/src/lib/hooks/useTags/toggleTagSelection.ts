interface ToggleTagSelectionParams {
    /** The ID of the tag to toggle. */
    tagId: string;
    /** The active dataset collection ID whose selection is updated. */
    collectionId: string;
    /** The current set of selected tag IDs for this collection. */
    currentSelected: Set<string>;
    /** Updates the persisted selection for the collection after the toggle. */
    updateSelected: (collectionId: string, newSelected: Set<string>) => void;
}

export function toggleTagSelection({
    tagId,
    collectionId,
    currentSelected,
    updateSelected
}: ToggleTagSelectionParams): void {
    const newSelected = new Set(currentSelected);
    if (newSelected.has(tagId)) {
        newSelected.delete(tagId);
    } else {
        newSelected.add(tagId);
    }

    updateSelected(collectionId, newSelected);
}
