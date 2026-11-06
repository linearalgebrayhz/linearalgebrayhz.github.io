document.querySelectorAll('.courses-card').forEach(function (card) {
    var tabList = card.querySelector('.course-tabs');
    var tabs = Array.from(tabList.querySelectorAll('.course-tab'));
    var panels = tabs.map(function (tab) {
        return document.getElementById(tab.getAttribute('aria-controls'));
    });

    function selectTab(selectedIndex) {
        tabs.forEach(function (tab, index) {
            var selected = index === selectedIndex;
            tab.setAttribute('aria-selected', String(selected));
            tab.tabIndex = selected ? 0 : -1;
            panels[index].hidden = !selected;
        });
    }

    tabs.forEach(function (tab, index) {
        tab.setAttribute('role', 'tab');
        panels[index].setAttribute('role', 'tabpanel');
        panels[index].setAttribute('aria-labelledby', tab.id);
        panels[index].tabIndex = 0;

        tab.addEventListener('click', function () {
            selectTab(index);
        });

        tab.addEventListener('keydown', function (event) {
            var nextIndex;
            switch (event.key) {
                case 'ArrowRight': nextIndex = (index + 1) % tabs.length; break;
                case 'ArrowLeft': nextIndex = (index - 1 + tabs.length) % tabs.length; break;
                case 'Home': nextIndex = 0; break;
                case 'End': nextIndex = tabs.length - 1; break;
                default: return;
            }
            event.preventDefault();
            selectTab(nextIndex);
            tabs[nextIndex].focus();
        });
    });

    selectTab(0);
    tabList.setAttribute('role', 'tablist');
    card.classList.add('courses-enhanced');
    tabList.hidden = false;
});
