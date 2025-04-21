//keyboard$.subscribe(function(key) {
//  if (key.mode === "global" && key.type === "x") {
//    /* Add custom keyboard handler here */
//    key.claim()
//  }
//})


keyboard$.subscribe(function(key) {
  // CMD+K on Mac or CTRL+K on other platforms
  if (
    key.mode === "global" &&
    key.type === "k" &&
    (key.metaKey || key.ctrlKey)
  ) {
    key.claim()

    // Find and focus the search input
    const search = document.querySelector("input.md-search__input")
    if (search) {
      search.focus()
      search.select()
    }
  }
})
