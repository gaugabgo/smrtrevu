import click
import pandas as pd
import os
import datamapplot
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns


@click.command()
@click.option('--input_csv', required=True, help='Input CSV with topic assignments (output from topic modeling)')
@click.option('--model_path', required=True, help='Path to saved BERTopic model')
@click.option('--output_dir', default="visualizations", help='Directory for visualizations')
@click.option('--embeddings_2d_path', required=True, help='Path to pre-computed 2D embeddings (.npy file). Use "revu model reduce" to create these.')
@click.option('--metadata_csv', default=None, help='Path to separate metadata CSV file with additional fields for hover text (e.g., TI, AU, DP, AB, DOI, citation_count)')
@click.option('--label_mode', default='custom', type=click.Choice(['auto', 'custom']), help='Label mode: "auto" uses MMR topic words, "custom" uses the predefined custom_labels dict')
def create_visualizations(input_csv, model_path, output_dir, embeddings_2d_path, metadata_csv, label_mode):
    """
    Create visualizations from BERTopic model and topic assignments
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)

    # Validate required columns
    required_cols = ['processed_text', 'topic', 'probability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Make sure to run topic modeling first.")

    texts = df['processed_text'].astype(str).tolist()
    topics = df['topic'].tolist()
    print(f"Loaded {len(texts)} documents with topic assignments")

    print(f"Loading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(model_path)
    print("BERTopic model loaded")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Creating visualizations...")
    print("="*50 + "\n")

    # Auto labels: derived from MMR topic aspects
    custom_labels = {
        -1: "Outlier",
        0: "Ecosystem Ecology",
        1: "Causality in Physics",
        2: "Agriculture, Food Security & Climate Change",
        3: "Philosophy of Causality",
        4: "Energy Consumption & Environmental Causality",
        5: "Brain Connectivity & Granger Causality",
        6: "Mendelian Randomization in Cancer & Genetics",
        7: "Structural Equation Modeling & Latent Variable Models",
        8: "Tourism Satisfaction & Sustainable Tourism",
        9: "Technology Acceptance & Online Learning",
        10: "Entrepreneurial Innovation & Knowledge Management",
        11: "Propensity Score Methods & Average Treatment Effect Estimation",
        12: "Trade, Economic Growth, Foreign Investment & Granger Causality",
        13: "Green Innovation, Sustainability & Green Supply Chain Management",
        14: "Inflation, Exchange Rates, Stock Markets & Monetary Policy",
        15: "DAG-Based Task Scheduling & Parallel Computing",
        16: "Regression Discontinuity Design in Education & School Achievement",
        17: "-1",
        18: "Medicaid/Medicare Expansion using Difference in Difference Methods",
        19: "Sports Participation, Physical Activity & Motivational Climate",
        20: "Entrepreneurial Intention, Education, and Attitudes",
        21: "Academic Achievement, Self-Efficacy & Motivation",
        22: "Brand Image, Purchase Intention & Consumer Marketing",
        23: "Travel Behavior, Urban Mobility & Autonomous Vehicles",
        24: "Obesity, Eating Disorders & Body Image",
        25: "-1",
        26: "Supply Chain Management",
        27: "Employee Performance, Job Satisfaction & Organizational Culture",
        28: "Construction Project Management & Project Success",
        29: "Directed Acyclic Graphs",
        30: "Mobile Banking, Digital Payments & FinTech Adoption",
        31: "Green Purchase Intention, Eco-Consumption & Green Marketing",
        32: "Carbon Emissions, Air Pollution & Environmental Policy using Difference in Difference Methods",
        33: "Alcohol & Substance Use Disorders in Adolescents",
        34: "Literacy, Fluency & Second Language Acquisition",
        35: "Economics of Fertility and Parental Leave",
        36: "Subspace Identification using Instrumental Variable Techniques",
        37: "Gut Microbiome: Mendelian Randomization Studies",
        38: "Lung Cancer Surgery & Survival",
        39: "Housing Markets",
        40: "Teacher Performance and Job Satisfaction",
        41: "Occupational Safety",
        42: "Electoral Politics, Voter Turnout & Regression Discontinuity",
        43: "Organic Food Purchase Intention & Health Consciousness",
        44: "Instrumental Variable Estimation",
        45: "Bayesian Networks for Fault Diagnosis & Root Cause Analysis",
        46: "Internet & Smartphone Addiction in Adolescents",
        47: "COVID-19 Pharmacoepidemiological Propensity Score Studies",
        48: "Nurse Job Satisfaction, Burnout & Work Engagement",
        49: "Road Traffic Safety",
        50: "Natural Disasters",
        51: "Telemedicine & Mobile Health Technology Acceptance",
        52: "Orthopedic Surgery: Knee & Hip Arthroplasty Outcomes",
        53: "HIV: Treatment Adherence & Prevention Behaviors",
        54: "Parenting & Child Emotional Regulation",
        55: "Environmental Regulation & Green Finance",
        56: "Labour Market Policies",
        57: "Policing, Violence & Neighborhood Crime Rates",
        58: "Customer Satisfaction & Service Quality",
        59: "Hematological Malignancies",
        60: "Smoking Cessation & Smoking during Pregnancy",
        61: "Prostate Cancer Surgery",
        62: "Tourism-Led Economic Growth & Granger Causality",
        63: "Brand Community & Loyalty",
        64: "Gastric Cancer Surgery & Survival using Propensity Score Methods",
        65: "Blockchain & DAG-Based Distributed Ledger Technologies",
        66: "Adverse Drug Reactions & Pharmacovigilance",
        67: "Intimate Partner Violence & Sexual Violence",
        68: "Cardiac Valve Surgery",
        69: "Breast Cancer Surgery & Survival using Propensity Score Methods",
        70: "Internal Audit Quality & Fraud Prevention",
        71: "AI Literacy & Generative AI in Education",
        72: "Corporate Social Responsibility",
        73: "Pancreatic Cancer Surgery & Outcomes using Propensity Score Methods",
        74: "Sleep Quality, Insomnia & Mental Health",
        75: "Hepatocellular Carcinoma: Treatment & Survival",
        76: "Diabetes Pharmacotherapy",
        77: "Oil Prices, Stock Markets & Granger Causality",
        78: "Air Pollution & Environmental Health Effects",
        79: "Esophageal Cancer Surgery using Propensity Score Methods",
        80: "Waste Management & Recycling ",
        81: "Granger Causality Testing & Time Series Analysis",
        82: "Mendelian Randomization: Genetic Variants & GWAS",
        83: "Gene Regulatory Networks & Single-Cell Expression",
        84: "Coronary Artery Disease",
        85: "Sport Sponsorship & Fan Behavior",
        86: "Impulse Buying & Live-Streaming Commerce",
        87: "Petri Nets & Concurrent Systems",
        88: "Financial Literacy & Behavior",
        89: "Epigenetics & Occupational Exposure to Carcinogens",
        90: "Intensive Care, Sepsis & ICU Mortality",
        91: "Pro-Environmental Behavior & Attitudes",
        92: "Dairy & Livestock Production",
        93: "Suicidal Ideation & Prevention in Adolescents",
        94: "Religiosity, Spirituality & Mental Health",
        95: "Quality of Life & Psychosocial Outcomes in Cancer Patients",
        96: "Liver & Kidney Transplant Outcomes using Propensity Score Methods",
        97: "Banking Efficiency, Liquidity & Financial Stability",
        98: "Halal Products & Muslim Consumer Behavior",
        99: "Incarceration & Criminal Justice",
        100: "Insulin Signaling & Insulin Resistance",
        101: "Causal Discovery & Causal Graph Learning",
        102: "Event Causality Extraction & Natural Language Processing",
        103: "Chronic Pain & Fibromyalgia ",
        104: "Ischemic Stroke",
        105: "Dialysis Treatment and Mortality using Propensity Score Methods",
        106: "Oral Health & Dental Anxiety",
        107: "Customer Loyalty & Service Quality",
        108: "Digital Transformation & Dynamic Capabilities",
        109: "High-Speed Railway & Regional Development using Difference in Difference Methods",
        110: "Epidemiological Causal Inference & Public Health",
        111: "Bayesian Network Learning & Graphical Models",
        112: "Mendelian Randomization in Neurological & Psychiatric Disorders",
        113: "Leadership Styles, Ethical Leadership & Employee Creativity",
        114: "PTSD ",
        115: "Islamic Banking & FinTech Adoption in Indonesia",
        116: "Retirement, Pension Reform & Health Effects using Regression Discontinuity Designs",
        117: "Spinal Surgery and Outcomes using Propensity Score Methods",
        118: "Renewable Energy Adoption & Energy Saving Behavior",
        119: "Patient Satisfaction & Healthcare Service Quality",
        120: "Renal Cell Carcinoma Nephrectomy & Outcomes using Propensity Score Methods",
        121: "Electric Vehicle Adoption Intention",
        122: "Corporate Governance, CEO Compensation & Firm Performance",
        123: "Career Adaptability, Employability & Career Development",
        124: "Hepatitis, Antiviral Therapy & Risk",
        125: "Implicit Causality in Linguistics: Pronouns & Discourse",
        126: "Information and Communication Technology, Internet Access & Economic Growth",
        127: "Hepatocellular Carcinoma Surgery & Outcomes using Propensity Score Methods",
        128: "Immigration, Migration & Labour Market Integration",
        129: "Thyroid Cancer Surgery and Outcomes using Propensity Score Methods",
        130: "Regression Discontinuity Design: Methods & Inference",
        131: "Urban Green Space, Parks & Mental Health",
        132: "Corruption, Anti-Corruption Policies & Economic Growth",
        133: "Blockchain Technology Adoption",
        134: "Vaccine Hesitancy & Vaccination Intention",
        135: "Ophthalmic Surgery: Cataract, Glaucoma & Retinal Disorders",
        136: "Working Memory, Executive Function & Cognitive Ability",
        137: "Corporate Social Responsibility & Customer Loyalty",
        138: "Tax Compliance & Tax Evasion",
        139: "Statin Therapy & Cardiovascular Outcomes",
        140: "IoT, Mobile Commerce & Smart Home Adoption",
        141: "Total Quality Management & Organizational Performance",
        142: "Online Shopping, Trust & Repurchase Intention",
        143: "Mindfulness, Gratitude & Psychological Well-being",
        144: "Coronary Stenting Methods and Outcomes",
        145: "Health Expenditure, Economic Growth & Population Health",
        146: "Childbirth Delivery Types & Outcomes using Propensity Score Methods",
        147: "Agricultural Commodity Prices, Cointegration & Granger Causality",
        148: "Student Satisfaction & University Service Quality",
        149: "Financial Literacy, Investment Decisions & Behavioral Biases",
        150: "Drug-Induced Liver Injury & Causality Assessment",
        151: "Oral Anticoagulants: Warfarin vs. DOACs in Atrial Fibrillation",
        152: "Causal Mediation Analysis & Moderated Mediation",
        153: "Inflammatory Bowel Disease: Crohn's, Colitis & Biologic Therapies",
        154: "Work Engagement, Organizational Commitment & Perceived Support",
        155: "Antipsychotic Medications & Schizophrenia Treatment",
        156: "Gallbladder & Biliary Surgery",
        157: "International Remittances, Migration & Poverty Reduction",
        158: "IVF, Embryo Transfer & Reproductive Outcomes",
        159: "Fuzzy Cognitive Maps",
        160: "Board Gender Diversity & Firm Performance",
        161: "Atrial Fibrillation: Catheter Ablation",
        162: "Emotional Intelligence, Leadership & Employee Performance",
        163: "Maternal & Neonatal Health: Antenatal Care",
        164: "Human Resource Management & Organizational Performance",
        165: "Online Gaming, Gamification & Purchase Intention",
        166: "Head & Neck Cancer: Radiotherapy & Chemotherapy",
        167: "Causal Reasoning in Child Development",
        168: "Work-Family Conflict & Life Balance",
        169: "Accounting Information Systems & Financial Reporting",
        170: "Cannabis Legalization, Medical Use & Psychosis Risk",
        171: "Augmented/Virtual Reality in Tourism",
        172: "Childhood Asthma, Allergies & Atopic Dermatitis",
        173: "Cybersecurity Behavior & Information Security Compliance",
        174: "Bladder Cancer: Radical Cystectomy",
        175: "Social & Political Trust, Civic Engagement",
        176: "IoT Routing Protocols & Sensor Networks",
        177: "Cryptocurrency Markets & Bitcoin Price Volatility",
        178: "Workplace Spirituality & Spiritual Leadership",
        179: "Organizational Agility & Dynamic Capabilities",
        180: "Personality Traits & Psychometric Assessment",
        181: "Donation Behavior & Charitable Giving",
        182: "Grundtvig, Folk High Schools & Danish Theology",
        183: "Diabetes Self-Management & Glycemic Control",
        184: "Colorectal Cancer Chemotherapy & Outcomes using Propensity Score Methods",
        185: "Dividend Policy, Capital Structure & Firm Value",
        186: "Colorectal Surgery & Outcomes using Propensity Score Methods",
        187: "Multiple Sclerosis: Disease-Modifying Therapies",
        188: "Microfinance, Women Empowerment & Poverty Reduction",
        189: "Migraine ",
        190: "Opioid Prescribing & Opioid Use Disorder",
        191: "Postpartum Depression & Maternal Mental Health",
        192: "E-Government Adoption & Digital Public Services",
        193: "AI Chatbots & Customer Service",
        194: "Student Creativity & Creative Self-Efficacy",
        195: "Elderly Mental Health, Social Participation & Life Satisfaction",
        196: "Environmental, Social, and Governance Disclosure",
        197: "Loneliness, Social Isolation & Depression in Older Adults",
        198: "AI Adoption in Human Resource Management & Recruitment",
        199: "Peacekeeping Operations, Conflict & Political Science",
        200: "Cloud Computing Adoption & SaaS",
        201: "ADHD: Diagnosis & Interventions in Children & Adults",
        202: "Big Data Analytics & Supply Chain Strategy",
        203: "COPD & Respiratory Outcomes",
        204: "Rheumatoid Arthritis: TNF Inhibitors & Methotrexate",
        205: "Luxury Brand Consumption ",
        206: "Online Food Delivery & Customer Satisfaction",
        207: "Lean Manufacturing & Operational Performance",
        208: "Causal Inference in Recommender Systems",
        209: "Renin-Angiotensin System & Antihypertensive Therapy",
        210: "Military Expenditure, Defense Spending & Economic Growth",
        211: "Cardiac Arrest, CPR & Resuscitation Outcomes",
        212: "DAGs in Clinical & Epidemiological Contexts",
        213: "School Bullying & Victimization",
        214: "Advertising Effectiveness & Ad Avoidance",
        215: "Multisensory Perception & Bayesian Causal Inference",
        216: "Marital Satisfaction & Romantic Relationships",
        217: "Cyber Attacks, Intrusion Detection & Malware",
        218: "Proton Pump Inhibitors & Gastric Cancer Risk using Propensity Score Methods",
        219: "Cervical & Ovarian Cancer Surgery Survival using Propensity Score Methods",
        220: "Brain Tumor Surgery & Survival using Propensity Score Methods",
        221: "Organizational Citizenship Behavior & Servant Leadership",
        222: "Enterprise Resource Planning Impacts",
        223: "Rectal Cancer Surgery & Outcomes using Propensity Score Methods",
        224: "Terrorism & Fear of Terror",
        225: "Music Education & Self-Efficacy",
        226: "Stigma & Help-Seeking Behavior",
        227: "Vaccine Safety & Adverse Events Following Immunization",
        228: "-1",
        229: "Materials Science: Alloys, Coatings & Diffusion",
        230: "Online Privacy & Self-Disclosure",
        231: "Academic Procrastination & Self-Regulation",
        232: "Problem Gambling, Gaming Addiction & Loot Boxes",
        233: "Hernia Repair Surgery",
        234: "Solar-Terrestrial Physics & Geomagnetic Storms",
        235: "Volunteering, Volunteer Motivation & Social Capital",
        236: "Family Business Governance & Succession Planning",
        237: "Intellectual Property & Patent Policy",
        238: "Sleep Disorders & Mendelian Randomization",
        239: "Diabetes Disease Management Programs using Difference in Difference Methods",
        240: "Helicobacter Pylori & Gastric Ulcer Treatment",
        241: "Quantile Regression & Instrumental Variable Quantile Methods",
        242: "Perfectionism: Adaptive vs. Maladaptive",
        243: "Intrahepatic Cholangiocarcinoma Surgery & Prognosis using Propensity Score Methods",
        244: "Cyberbullying & Adolescent Cyber Victimization",
        245: "COVID-19 Pandemic Policies & Epidemic Control",
        246: "Juvenile Delinquency & Peer Influence",
        247: "Counterfactual Reasoning & Causal Inference",
        248: "Endoscopic Submucosal Dissection (ESD)",
        249: "AI-Driven Personalization & Digital Marketing",
        250: "Religiosity, Secularization & Sociology of Religion",
        251: "Breastfeeding Practices",
        252: "Child Malnutrition, Stunting & Feeding Practices",
        253: "Employee Turnover Intention & Job Satisfaction",
        254: "Service Robots & Robot Acceptance in Hospitality",
        255: "Causal Inference Methods & Big Data",
        256: "Hearing Loss, Tinnitus & Vestibular Disorders",
        257: "Dynamic Panel Data Models: GMM & Instrumental Variables",
        258: "Malaria Prevention & Chemoprevention",
        259: "Food Tourism & Culinary Tourism",
        260: "Legal Causation, Medical Malpractice & Liability",
        261: "Social Anxiety, Anxiety Disorders & Emotion Regulation",
        262: "Fake News, Journalism & Social Media",
        263: "Support Vector Machines, Multiclass Classification & DAGs",
        264: "Organizational Citizenship Behavior & Organizational Justice",
        265: "Vaccine-Associated Adverse Neurological Events",
        266: "Digital Library Services & User Satisfaction",
        267: "Education Expenditure, Economic Growth & Granger Causality",
        268: "Algorithmic, Counterfactual Fairness & Discrimination",
        269: "Schizophrenia, Neurocognition & Social Functioning",
        270: "Innovative Work Behavior & Transformational Leadership",
        271: "Environmental Kuznets Curve & Ecological Footprint",
        272: "Alzheimer's Disease: Neurodegeneration & Amyloid Pathology",
        273: "Intracranial Aneurysm Treatments & Outcomes using Propensity Score Methods",
        274: "Colorectal Liver Metastasis Surgery Outcomes using Propensity Score Methods",
        275: "Aortic Aneurysm & Endovascular Repair",
        276: "Hume's Philosophy of Causality",
        277: "Sustainable Fashion & Apparel Consumption",
        278: "Bond Graph Modeling & Causality in Physical Systems",
        279: "Digital Leadership & Employee Digital Competence",
        280: "Phylogenetic Networks & Evolutionary Genomics",
        281: "Causality & Econometrics",
        282: "-1",
        283: "Vocational Rehabilitation, Compulsory Schooling & Employment",
        284: "Pulmonary Embolism & Venous Thromboembolism",
        285: "Periodontal Disease, Tooth Loss & Obesity",
        286: "Osteoporosis Treatment: Bisphosphonates & Fracture Prevention",
        287: "Tuberculosis Treatment & MDR-TB",
        288: "Wine Tourism & Wine Consumer Behavior",
        289: "Dementia Risk & Alzheimer's Prevention",
        290: "Bariatric Surgery & Weight Loss",
        291: "Causal Representation Learning & Domain Generalization",
        292: "Neural Architecture Search & Deep Neural Networks",
        293: "Kantian Philosophy of Causality",
        294: "Urological Stone Disease: Nephrolithotomy & Ureteroscopy",
        295: "Internet Use, Social Participation & Depression in Elderly",
        296: "Difference-in-Differences: Parallel Trends & Staggered Adoption",
        297: "Teacher Burnout & Academic Stress",
        298: "Venture Capital Investment & Innovation",
        299: "Mergers & Acquisitions: Market Effects & Regulation",
        300: "Gene Ontology & DAG-Based Ontologies"
    }

    if label_mode == 'auto':
        mmr_aspects = topic_model.topic_aspects_.get("MMR", {})
        document_labels = [
            ", ".join([word for word, _ in mmr_aspects[topic][:3]])
            if topic != -1 and topic in mmr_aspects else "Outlier"
            for topic in topics
        ]
    else:
        document_labels = [custom_labels.get(topic, f"Topic {topic}") for topic in topics]
    print(f"Label mode: {label_mode}")
    print(f"Number of mapped labels: {len(document_labels)}")
    print(f"First few mapped labels: {document_labels[:5]}")


    # ========================================
    # LOAD PRE-COMPUTED 2D EMBEDDINGS
    # ========================================
    print(f"Loading pre-computed 2D embeddings from {embeddings_2d_path}...")
    if not os.path.exists(embeddings_2d_path):
        raise FileNotFoundError(
            f"2D embeddings file not found: {embeddings_2d_path}\n"
            f"Please compute 2D embeddings first using:\n"
            f"  revu model reduce --embeddings_path <path> --output_path {embeddings_2d_path}"
        )

    embeddings_2d = np.load(embeddings_2d_path)
    print(f"✓ Loaded 2D embeddings with shape: {embeddings_2d.shape}")

    # Validate embeddings match document count
    if len(embeddings_2d) != len(texts):
        raise ValueError(
            f"2D embeddings count ({len(embeddings_2d)}) doesn't match document count ({len(texts)}). "
            f"Please ensure the embeddings were computed from the same dataset."
        )

    # ========================================
    # BERTOPIC BUILT-IN VISUALIZATIONS
    # ========================================
    print("\n1. Creating BERTopic barchart...")
    fig = topic_model.visualize_barchart(top_n_topics=len(set(topics)))
    fig.write_html(os.path.join(output_dir, "CausalInference_barchart.html"))
    print("   ✓ Saved BERTopic_barchart.html")

    print("2. Creating BERTopic topics visualization...")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(output_dir, "CausalInference_BERTopic_Topics.html"))
    print("   ✓ Saved BERTopic_Topics.html")

    print("3. Creating BERTopic heatmap...")
    fig_topics_heatmap = topic_model.visualize_heatmap()
    fig_topics_heatmap.write_html(os.path.join(output_dir, "CausalInference_BERTopic_Heatmap.html"))
    print("   ✓ Saved BERTopic_Heatmap.html")

    # ========================================
    # DATAMAPPLOT STATIC VISUALIZATION
    # ========================================
    
    print("\n4. Creating static datamapplot (outliers filtered)...")

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'topic': topics,
        'label': document_labels,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
    })

    # Filter out outliers (topic -1)
    filtered_df = plot_df[plot_df['topic'] != -1]

    # Subsample for memory efficiency if dataset is large
    max_static_points = 50_000
    if len(filtered_df) > max_static_points:
        print(f"   Subsampling {len(filtered_df)} points to {max_static_points} for static plot...")
        filtered_df = filtered_df.groupby('topic', group_keys=False).apply(
            lambda g: g.sample(frac=max_static_points / len(filtered_df), random_state=42)
        ).reset_index(drop=True)
        print(f"   ✓ Subsampled to {len(filtered_df)} points")

    filtered_embeddings = filtered_df[['x', 'y']].values
    filtered_labels = filtered_df['label'].tolist()

    fig_docs, ax_docs = datamapplot.create_plot(
        filtered_embeddings,
        filtered_labels,
        title="Topics in the Causal Inference Literature",
        sub_title="Data map of BERTopic-extracted topics",
        label_over_points=False,
        dynamic_label_size=True,
        label_font_size=8,
        label_wrap_width=20,
        figsize=(16, 12)
    )
    static_plot_path = os.path.join(output_dir, "CausalInference_BERTopic_dmp_static.png")
    fig_docs.savefig(static_plot_path, bbox_inches="tight", dpi=100)
    print(f"   ✓ Saved {static_plot_path}")

    # ========================================
    # DATAMAPPLOT INTERACTIVE VISUALIZATION
    # ========================================
    print("\n5. Creating interactive datamapplot...")

    # ========================================
    # LOAD METADATA AND CREATE HOVER TEXT
    # ========================================
    hover_texts = []
    publication_years = None

    if metadata_csv and os.path.exists(metadata_csv):
        print(f"Loading metadata from {metadata_csv}...")
        metadata_df = pd.read_csv(metadata_csv, dtype=str).fillna('')

        # Find ID column in both dataframes (case-insensitive)
        topic_id_col = None
        metadata_id_col = None

        for col in df.columns:
            if col.lower() == 'id':
                topic_id_col = col
                break

        for col in metadata_df.columns:
            if col.lower() == 'id':
                metadata_id_col = col
                break

        if topic_id_col and metadata_id_col:
            # Merge metadata with topic data on ID
            df_with_metadata = df.merge(
                metadata_df,
                left_on=topic_id_col,
                right_on=metadata_id_col,
                how='left',
                suffixes=('', '_meta')
            )
            print(f"✓ Merged metadata for {len(df_with_metadata)} documents")

            # If merge inflated rows (duplicate IDs in metadata), fall back to df directly
            if len(df_with_metadata) != len(df):
                print(f"  Warning: merge produced {len(df_with_metadata)} rows vs {len(df)} documents — using input data directly for hover text")
                df_with_metadata = df.reset_index(drop=True)
            else:
                df_with_metadata = df_with_metadata.reset_index(drop=True)

            # Create rich hover text with metadata fields
            hover_texts = []
            for pos, row in df_with_metadata.iterrows():
                hover_parts = [f"Topic: {document_labels[pos]}"]

                # Add available metadata fields
                if 'title' in row and row['title']:
                    hover_parts.append(f"Title: {row['title']}")
                if 'authorships.raw_author_name' in row and row['authorships.raw_author_name']:
                    hover_parts.append(f"Authors: {row['authorships.raw_author_name']}")
                if 'publication_year' in row and row['publication_year']:
                    hover_parts.append(f"Year: {row['publication_year']}")
                if 'cited_by_count' in row and row['cited_by_count']:
                    hover_parts.append(f"Citation Count: {row['cited_by_count']}")
                if 'DOI' in row and row['DOI']:
                    hover_parts.append(f"Link: https://doi.org/{row['DOI']}")

                hover_texts.append('\n'.join(hover_parts))

    fig_dmp = datamapplot.create_interactive_plot(
        embeddings_2d,
        document_labels,
        hover_text=hover_texts,
        enable_search=True,
        title="Document Topics",
        sub_title="Interactive data map of BERTopic-extracted topics",
        noise_label="Outlier",
        histogram_data=publication_years,
        initial_zoom_fraction=0.9,
    )
    interactive_plot_path = os.path.join(output_dir, "BERTopic_interactive_dmplot.html")
    fig_dmp.save(interactive_plot_path)
    print(f"   ✓ Saved {interactive_plot_path}")

    # ========================================
    # TOPIC PROPORTIONS BY YEAR VISUALIZATION
    # ========================================
    if metadata_csv and os.path.exists(metadata_csv):
        print("\n6. Creating topic proportions by year visualization...")

        # Load metadata if not already loaded
        if 'df_with_metadata' not in locals():
            metadata_df = pd.read_csv(metadata_csv, dtype=str).fillna('')

            # Find ID column in both dataframes (case-insensitive)
            topic_id_col = None
            metadata_id_col = None

            for col in df.columns:
                if col.lower() == 'id':
                    topic_id_col = col
                    break

            for col in metadata_df.columns:
                if col.lower() == 'id':
                    metadata_id_col = col
                    break

            if topic_id_col and metadata_id_col:
                df_with_metadata = df.merge(
                    metadata_df,
                    left_on=topic_id_col,
                    right_on=metadata_id_col,
                    how='left',
                    suffixes=('', '_meta')
                )

        # Check if publication_year column exists
        if 'publication_year' in df_with_metadata.columns:
            # Convert publication_year to numeric, handling any non-numeric values
            df_with_metadata['publication_year'] = pd.to_numeric(
                df_with_metadata['publication_year'],
                errors='coerce'
            )

            # Filter out rows with invalid/missing years
            df_valid_years = df_with_metadata[df_with_metadata['publication_year'].notna()].copy()

            # (1) Total publications by year
            total_pubs_by_year = df_valid_years.groupby('publication_year').size()
            print(f"   → Found publications across {len(total_pubs_by_year)} years")

            # (2) Count topics per year
            topic_counts_by_year = df_valid_years.groupby(['publication_year', 'topic']).size().reset_index(name='count')

            # (3) Calculate proportions (topic publications / total publications * 100)
            topic_proportions = topic_counts_by_year.copy()
            topic_proportions['total_pubs'] = topic_proportions['publication_year'].map(total_pubs_by_year)
            topic_proportions['proportion'] = (topic_proportions['count'] / topic_proportions['total_pubs']) * 100

            # Create the plot
            unique_topics = sorted(df_valid_years['topic'].unique())

            # Filter out outliers (topic -1) from visualization if desired
            topics_to_plot = [t for t in unique_topics if t != -1]

            # Set up color palette
            n_topics = len(topics_to_plot)
            colors = sns.color_palette("husl", n_topics)

            plt.figure(figsize=(14, 8))

            for idx, topic_id in enumerate(topics_to_plot):
                topic_data = topic_proportions[topic_proportions['topic'] == topic_id]

                # Get topic label
                topic_label_list = topic_model.get_topic(topic_id)
                if topic_label_list:
                    topic_label = f"Topic {topic_id}: {', '.join([word for word, _ in topic_label_list[:3]])}"
                else:
                    topic_label = f"Topic {topic_id}"

                plt.plot(
                    topic_data['publication_year'],
                    topic_data['proportion'],
                    marker='o',
                    label=topic_label,
                    color=colors[idx],
                    linewidth=2,
                    markersize=4
                )

            plt.xlabel('Publication Year', fontsize=12)
            plt.ylabel('Topic Proportion (%)', fontsize=12)
            plt.title('Topic Proportions Over Time\n(Topic Publications as % of Total Yearly Publications)', fontsize=14, pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(output_dir, "topic_proportions_by_year.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved {plot_path}")

            # (4) Append yearly proportion values to topic CSV
            # Pivot the data to have years as columns
            proportion_pivot = topic_proportions.pivot(
                index='topic',
                columns='publication_year',
                values='proportion'
            ).fillna(0)

            # Rename columns to indicate they're proportions
            proportion_pivot.columns = [f'proportion_year_{int(year)}' for year in proportion_pivot.columns]
            proportion_pivot = proportion_pivot.reset_index()

            # Load the original topic CSV to append to
            df_output = df.copy()

            # Merge the proportion data with the main dataframe
            df_output = df_output.merge(proportion_pivot, on='topic', how='left')

            # Save updated CSV
            output_csv_path = os.path.join(output_dir, "topics_with_yearly_proportions.csv")
            df_output.to_csv(output_csv_path, index=False)
            print(f"   ✓ Saved updated CSV with yearly proportions: {output_csv_path}")

            # Also create a summary CSV with just topic-level statistics
            topic_summary = topic_proportions.groupby('topic').agg({
                'count': 'sum',
                'proportion': 'mean'
            }).reset_index()
            topic_summary.columns = ['topic', 'total_publications', 'avg_yearly_proportion']

            # Add topic labels
            topic_summary['topic_label'] = topic_summary['topic'].apply(
                lambda t: ', '.join([word for word, _ in topic_model.get_topic(t)[:5]])
                if topic_model.get_topic(t) else 'Outlier'
            )

            # Merge with yearly proportions
            topic_summary = topic_summary.merge(proportion_pivot, on='topic', how='left')

            summary_csv_path = os.path.join(output_dir, "topic_yearly_proportions_summary.csv")
            topic_summary.to_csv(summary_csv_path, index=False)
            print(f"   ✓ Saved topic summary with yearly proportions: {summary_csv_path}")

        else:
            print("   ⚠ Warning: 'publication_year' column not found in metadata. Skipping yearly proportion visualization.")
    else:
        print("\n⚠ Skipping topic proportions by year visualization (no metadata CSV provided)")

    print("\n" + "="*50)
    print("✅ All visualizations created successfully!")
    print("="*50)

    return embeddings_2d
