import { useGlobalStorage } from '$lib/hooks';

type PanelType = Parameters<ReturnType<typeof useGlobalStorage>['setActivePanel']>[0];

export function useSidePanelTabs() {
    const { activePanel, setActivePanel } = useGlobalStorage();

    function toggle(activePanel: PanelType | 'none', panel: PanelType) {
        setActivePanel(activePanel === panel ? 'none' : panel);
    }

    return { activePanel, toggle };
}
