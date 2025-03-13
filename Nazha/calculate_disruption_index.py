import pandas as pd

def calculate_disruption_index(focal_paper, citing_papers, references, full_data):
    NF = 0
    NB = 0
    NR = 0

    for citing_paper in citing_papers:
        citing_references = full_data[full_data['citing_paper_title'] == citing_paper]['citing_paper_references'].values
        if len(citing_references) == 0:
            continue

        citing_references = citing_references[0]
        if not isinstance(citing_references, str):
            citing_references = ""

        citing_references_list = citing_references.split(';')

        # Debugging information for citing paper and references
        print(f"DEBUG - Citing Paper: {citing_paper}, References: {citing_references_list}")

        cites_focal_only = focal_paper in citing_references_list and not any(ref in citing_references_list for ref in references)
        cites_both = focal_paper in citing_references_list and any(ref in citing_references_list for ref in references)
        cites_refs_only = focal_paper not in citing_references_list and any(ref in citing_references_list for ref in references)

        if cites_focal_only:
            NF += 1
        if cites_both:
            NB += 1
        if cites_refs_only:
            NR += 1

    try:
        disruption_index = (NF - NB) / (NF + NB + NR)
    except ZeroDivisionError:
        disruption_index = 0

    print(f"DEBUG - focal: {focal_paper}, NF: {NF}, NB: {NB}, NR: {NR}, D: {disruption_index}")
    return disruption_index

# Load your CSV file with a different encoding
data = pd.read_csv('final.csv', encoding='ISO-8859-1', low_memory=False)

# Print column names to verify
print(data.columns)

# Adjust these lines based on the actual column names
focal_column = 'paper_title'
citing_column = 'citing_paper_title'
references_column = 'citing_paper_references'
authors_column = 'paper_authors'

# Filter for Katalin Karikó's papers
kariko_papers = data[data[authors_column].str.contains('Katalin', na=False)]

# Print the number of papers and some sample data to verify
print(f"Number of Katalin Karikó's papers: {kariko_papers.shape[0]}")
print(kariko_papers[[focal_column, authors_column]].head())

# Prepare the data
results = []

for focal_paper in kariko_papers[focal_column].unique():
    citing_papers = kariko_papers[kariko_papers[focal_column] == focal_paper][citing_column].unique()
    references = kariko_papers[kariko_papers[focal_column] == focal_paper][references_column].unique()
    references = references[0].split(';') if len(references) > 0 and isinstance(references[0], str) else []

    # Debugging information for focal paper and its references
    print(f"DEBUG - Focal Paper: {focal_paper}, References: {references}")

    # Calculate the disruption index
    di = calculate_disruption_index(focal_paper, citing_papers, references, data)
    results.append({'Focal_Paper': focal_paper, 'Disruption_Index': di})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('disruption_index_results.csv', index=False)
