from langchain.utilities import WikipediaAPIWrapper


class CustomWikipediaAPIWrapper(WikipediaAPIWrapper):

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        WIKIPEDIA_MAX_QUERY_LENGTH = 300
        page_titles = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])
        summaries = []
        for page_title in page_titles[:self.top_k_results]:
            wiki_page = self._fetch_page(page_title)
            if wiki_page is None:
                pass
            else:
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return summaries