from langchain.utilities import WikipediaAPIWrapper


class CustomWikipediaAPIWrapper(WikipediaAPIWrapper):

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        WIKIPEDIA_MAX_QUERY_LENGTH = 300
        page_titles = self.wiki_client.search(query[:WIKIPEDIA_MAX_QUERY_LENGTH])
        summaries = []
        print(page_titles)
        for page_title in page_titles[:self.top_k_results]:
            #             print(f'Processing "{page_title}"')
            wiki_page = self._fetch_page(page_title)
            if wiki_page is None:
                pass
            # #                 print(f'Could not fetch "{page_title}"')
            #             elif query.lower() not in wiki_page.content.lower():
            # #                 print(f'Query not found in "{page_title}"')
            else:
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return summaries