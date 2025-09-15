import sys
import re
import csv
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urljoin
import warnings
import difflib

def scrape_race_data(start_url, query_str):
    """
    Finds a specific race report from a starting URL, scrapes its data,
    and saves it to a CSV file.
    """
    try:
        # Filter the XMLParsedAsHTMLWarning
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        # 1. Fetch the main event page
        print(f"Fetching data from {start_url}...", file=sys.stderr)

        main_response = requests.get(start_url, timeout=10)
        main_response.raise_for_status()
        soup = BeautifulSoup(main_response.content, 'html.parser')

        # 2. Find all potential report links
        report_links = soup.find_all('a', class_='dropdown-item', onclick=True)
        found_report = None
        best_match_name = ""


        # 3. Collect all report links
        report_candidates = []
        for link in report_links:
            onclick_attr = link['onclick']
            match = re.search(r"doAjaxCall\('([^']*)',\s*'([^']*)'\)", onclick_attr)
            if match:
                report_candidates.append(match.groups())  # (url, name)

        if not report_candidates:
            print("No valid reports found on page. Make sure you paste the link/URL to a myrcm page for the event and class you are interested in. For example https://myrcm.ch/myrcm/report/en/87328/360726#", file=sys.stderr)
            return

        # Build query string from all CLI args after URL
        query_str = " ".join(sys.argv[2:]).strip()
        if not query_str:
            print("You must provide a search string after the URL.", file=sys.stderr)
            print("Example: python myrcm_scraper.py <url> Final E", file=sys.stderr)
            return

        # Tokenize & scoring
        def tokenize(s: str) -> set[str]:
            return set(re.findall(r'[a-z0-9]+', s.lower()))


        def score(query: str, candidate: str) -> float:
            q_tokens = tokenize(query)
            c_tokens = tokenize(candidate)
            if not q_tokens:
                return 0.0

            overlap = len(q_tokens & c_tokens)
            base_score = overlap / len(q_tokens)

            # Priority adjustments
            cand_lower = candidate.lower()
            if "practice" in cand_lower:
                base_score -= 0.2   # small penalty
            if "final" in cand_lower and "practice" not in cand_lower:
                base_score += 0.2   # small bonus

            return base_score


        # Pick best candidate
        best_match = max(report_candidates, key=lambda rc: score(query_str, rc[1]))
        found_report, best_match_name = best_match

        if score(query_str, best_match_name) == 0:
            msg = f"ERROR: No report matched '{query_str}'."
            print(msg, file=sys.stdout)   # send to stdout so runner can catch it
            print("Available reports:", file=sys.stderr)
            for _, name in report_candidates:
                print("  -", name, file=sys.stderr)
            return


        print(f"Found report: '{best_match_name}'", file=sys.stderr)

        #--
    
        race_query_for_filename = best_match_name.split(' :: ')[-1].strip()

        # 4. Scrape the specific report page
        report_url = urljoin(start_url, found_report)
        print(f"➡️  Scraping data from {report_url}...")
        report_response = requests.get(report_url, timeout=10)
        report_response.raise_for_status()
        report_soup = BeautifulSoup(report_response.content, 'html.parser')

        all_tables = report_soup.find_all("table")
        if not all_tables:
            print("Could not find any results tables on the report page.", file=sys.stderr)
            return

        # 5. Save the data to a CSV file
        title_tag = soup.find("title")
        full_event_title = ""
        abbr_title = ""
        if title_tag:
            full_event_title = title_tag.text.strip()
            title_parts = full_event_title.split(" :: ")
            event_title_part = title_parts[0].strip()
            abbr_title = "".join(word[0] for word in event_title_part.split() if word.isalnum()) + \
                         "".join(re.findall(r'\d+', event_title_part))
            
            if len(title_parts) > 1:
                class_part = title_parts[1].strip()
                abbr_title += "_" + re.sub(r'[^a-z0-9]+', '_', class_part.lower()).strip('_')

        clean_race_query = re.sub(r'[^a-z0-9]+', '_', race_query_for_filename.lower()).strip('_')
        csv_filename = f"{abbr_title}_{clean_race_query}.csv"

        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header information
            print("Writing header information...", file=sys.stderr)
            if full_event_title:
                writer.writerow([full_event_title])
            writer.writerow([best_match_name.split(' :: ')[-1].strip()])
            
            detailed_info_node = report_soup.find(string=re.compile(r"Section:\s*"))
            if detailed_info_node:
                writer.writerow([detailed_info_node.parent.get_text()])

            # Write the summary table
            print("Writing summary table...", file=sys.stderr)
            summary_table = all_tables[0]
            header = [th.text.strip() for th in summary_table.find_all("th")]
            writer.writerow(header)

            summary_rows = summary_table.find_all("tr")
            driver_names = []
            for row in summary_rows:
                cells = row.find_all("td")
                if cells:
                    driver_name = cells[3].text.strip()
                    driver_names.append(driver_name)
                    writer.writerow([cell.text.strip() for cell in cells])
            
            # --- Lap times ---
            if len(all_tables) > 1:
                writer.writerow(["Laptimes"])
                
                print("Processing and merging detailed lap data...", file=sys.stderr)
                
                all_laps = []
                headers = []
                max_laps = 0
                num_drivers = len(driver_names)
                
                for i, lap_table in enumerate(all_tables[1:]):
                    prev_element = lap_table.find_previous_sibling()
                    if prev_element:
                        text = prev_element.get_text(strip=True).lower()
                        if "records" in text or "corrections" in text:
                            print(f"Found '{prev_element.get_text(strip=True)}' section. Stopping laps data data writing.", file=sys.stderr)
                            break

                    # Get lap headers and data for the current driver's table
                    lap_header = [th.text.strip() for th in lap_table.find_all("th")]
                    lap_data = [[cell.text.strip() for cell in row.find_all("td")] for row in lap_table.find_all("tr") if row.find_all("td")]

                    # For all tables after the first, remove the redundant lap number column
                    if i > 0:
                        lap_header = lap_header[1:]
                        lap_data = [row[1:] for row in lap_data]

                    # Add this driver's header to our master headers list
                    #headers.extend(lap_header)
                    
                    # Store data in a list of lists for merging
                    if not all_laps:
                        # For the first table, initialize all_laps
                        all_laps = lap_data
                    else:
                        # For subsequent tables, merge with existing data
                        num_drivers_to_append = num_drivers - 10
                        if i > 0 and num_drivers_to_append > 0:
                            for j, row in enumerate(lap_data):
                                if j < len(all_laps):
                                    all_laps[j].extend(row[0:num_drivers_to_append])
                                else:
                                    # Handle case where a subsequent table has more laps
                                    # Pad with empty strings for the previous drivers
                                    num_previous_drivers_data_cols = len(headers) - (num_drivers_to_append)
                                    new_row = [''] * num_previous_drivers_data_cols
                                    new_row.extend(row[0:num_drivers_to_append])
                                    all_laps.append(new_row)

                    max_laps = max(max_laps, len(lap_data))
                
                # Write the header row with Laps + all driver names
                # Determine the target number of columns for consistency
                target_columns = max(10, num_drivers)
                while (len(driver_names) < target_columns):
                       driver_names.append("")

                writer.writerow(["Lap"] + driver_names)
                
                # Write the merged data rows
                # Keep track of accumulated times across laps
                accumulated_times = [0.0] * (len(all_laps[0]) - 1)  # one per driver

                # Write the merged data rows
                for row in all_laps:
                    lap_num = row[0]
                    lap_times = row[1:]

                    # Detect if positions already exist
                    if any(cell.strip().startswith("(") for cell in lap_times if cell.strip()):
                        # Already official → keep as-is
                        writer.writerow(row)
                    else:
                        # Update accumulated times
                        times_with_idx = []
                        for idx, cell in enumerate(lap_times):
                            if cell.strip():
                                try:
                                    if ":" in cell:
                                        parts = cell.split(":")
                                        if len(parts) == 2:
                                            minutes = int(parts[0])
                                            seconds = float(parts[1])
                                            lap_time_val = minutes * 60 + seconds
                                        else:
                                            raise ValueError(f"Unexpected lap time format: {cell}")
                                    else:
                                        lap_time_val = float(cell)

                                    accumulated_times[idx] += lap_time_val
                                    times_with_idx.append((accumulated_times[idx], idx, cell.strip()))

                                except ValueError as e:
                                    print(f"❌ Failed to parse lap time '{cell}' at driver {idx+1}, lap {lap_num}: {e}", file=sys.stderr)
                                    sys.exit(1)

                            else:
                                # No lap, keep accumulated time as-is (driver DNF or pit still counted)
                                pass

                        # Sort by accumulated total race time
                        times_with_idx.sort(key=lambda x: x[0])

                        # Assign positions
                        pos_map = {}
                        for pos, (acc_time, idx, lap_time_str) in enumerate(times_with_idx, start=1):
                            # Debug: include accumulated time in mm:ss format
                            minutes = int(acc_time // 60)
                            seconds = acc_time % 60
                            acc_time_str = f"{minutes:02d}:{seconds:05.2f}"
                            #pos_map[idx] = f"({pos}) {lap_time_str} [{acc_time_str}]"
                            pos_map[idx] = f"({pos}) {lap_time_str}"

                        # Rebuild row
                        new_row = [lap_num]
                        for idx, cell in enumerate(lap_times):
                            new_row.append(pos_map.get(idx, cell.strip()))

                        writer.writerow(new_row)

            else:
                writer.writerow([])

        print(csv_filename, file=sys.stdout)   # clean signal for wrapper script


    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python myrcm_scraper.py <url> <search terms...>")
        print("Example: python myrcm_scraper.py https://myrcm.ch/... Final A run 2")
        print("Example: python myrcm_scraper.py https://myrcm.ch/... 1/2 Final A")
        print("Example: python myrcm_scraper.py https://myrcm.ch/... Last Chance Final A 1")
        sys.exit(1)

    url_arg = sys.argv[1].strip()

    # Normalize: always force https://
    if url_arg.startswith("http://"):
        url_arg = "https://" + url_arg[len("http://"):]
    elif not url_arg.startswith("https://"):
        url_arg = "https://" + url_arg

    print(f"(ℹ️  Normalized URL to: {url_arg})", file=sys.stderr)

    query_str = " ".join(sys.argv[2:])
    scrape_race_data(url_arg, query_str)

