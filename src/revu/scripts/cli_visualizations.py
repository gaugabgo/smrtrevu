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
    print(f"✓ Loaded {len(texts)} documents with topic assignments")

    print(f"Loading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(model_path)
    print("✓ BERTopic model loaded")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("Creating visualizations...")
    print("="*50 + "\n")

    # Auto labels: derived from MMR topic aspects
    custom_labels = {
        -1: "-1",
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
        31: “-1”,
        32: “-1”,
        33: "Green Purchase Intention, Eco-Consumption & Green Marketing",
        34: "Carbon Emissions, Air Pollution & Environmental Policy using Difference in Difference Methods",
        35: "Alcohol & Substance Use Disorders in Adolescents",
        36: "Literacy, Fluency & Second Language Acquisition",
        37: "Economics of Fertility and Parental Leave",
        38: "Subspace Identification using Instrumental Variable Techniques",
        39: "Gut Microbiome: Mendelian Randomization Studies",
        40: "Lung Cancer Surgery & Survival",
        41: "Housing Markets",
        42: "Teacher Performance and Job Satisfaction",
        43: "Occupational Safety",
        44: "Electoral Politics, Voter Turnout & Regression Discontinuity",
        45: "Organic Food Purchase Intention & Health Consciousness",
        46: "Instrumental Variable Estimation",
        47: "Bayesian Networks for Fault Diagnosis & Root Cause Analysis",
        48: "Internet & Smartphone Addiction in Adolescents",
        49: "COVID-19 Pharmacoepidemiological Propensity Score Studies",
        50: "Nurse Job Satisfaction, Burnout & Work Engagement",
        51: "Road Traffic Safety",
        52: "Natural Disasters",
        53: "Telemedicine & Mobile Health Technology Acceptance",
        54: "Orthopedic Surgery: Knee & Hip Arthroplasty Outcomes",
        55: "HIV: Treatment Adherence & Prevention Behaviors",
        56: "Parenting & Child Emotional Regulation",
        57: "Environmental Regulation & Green Finance",
        58: "Labour Market Policies",
        59: "Policing, Violence & Neighborhood Crime Rates",
        60: "Customer Satisfaction & Service Quality",
        61: "Hematological Malignancies",
        62: "Smoking Cessation & Smoking during Pregnancy",
        63: "Prostate Cancer Surgery",
        64: "Tourism-Led Economic Growth & Granger Causality",
        65: "Brand Community & Loyalty",
        66: "Gastric Cancer Surgery & Survival using Propensity Score Methods",
        67: "Blockchain & DAG-Based Distributed Ledger Technologies",
        68: "Adverse Drug Reactions & Pharmacovigilance",
        69: "Intimate Partner Violence & Sexual Violence",
        70: "Cardiac Valve Surgery",
        71: "Breast Cancer Surgery & Survival using Propensity Score Methods",
        72: "Internal Audit Quality & Fraud Prevention",
        73: "AI Literacy & Generative AI in Education",
        74: "Corporate Social Responsibility",
        75: "Pancreatic Cancer Surgery & Outcomes using Propensity Score Methods",
        76: "Sleep Quality, Insomnia & Mental Health",
        77: "Hepatocellular Carcinoma: Treatment & Survival",
        78: "Diabetes Pharmacotherapy",
        79: "Oil Prices, Stock Markets & Granger Causality",
        80: "Air Pollution & Environmental Health Effects",
        81: "Esophageal Cancer Surgery using Propensity Score Methods",
        82: "Waste Management & Recycling ",
        83: "Granger Causality Testing & Time Series Analysis",
        84: "Mendelian Randomization: Genetic Variants & GWAS",
        85: "Gene Regulatory Networks & Single-Cell Expression",
        86: "Coronary Artery Disease",
        87: "Sport Sponsorship & Fan Behavior",
        88: "Impulse Buying & Live-Streaming Commerce",
        89: "Petri Nets & Concurrent Systems",
        90: "Financial Literacy & Behavior",
        91: "Epigenetics & Occupational Exposure to Carcinogens",
        92: "Intensive Care, Sepsis & ICU Mortality",
        93: "Pro-Environmental Behavior & Attitudes",
        94: "Dairy & Livestock Production",
        95: "Suicidal Ideation & Prevention in Adolescents",
        96: "Religiosity, Spirituality & Mental Health",
        97: "Quality of Life & Psychosocial Outcomes in Cancer Patients",
        98: "Liver & Kidney Transplant Outcomes using Propensity Score Methods",
        99: "Banking Efficiency, Liquidity & Financial Stability",
        100: "Halal Products & Muslim Consumer Behavior",
        101: "Incarceration & Criminal Justice",
        102: "Insulin Signaling & Insulin Resistance",
        103: "Causal Discovery & Causal Graph Learning",
        104: "Event Causality Extraction & Natural Language Processing",
        105: "Chronic Pain & Fibromyalgia ",
        106: "Ischemic Stroke",
        107: "Dialysis Treatment and Mortality using Propensity Score Methods",
        108: "Oral Health & Dental Anxiety",
        109: "Customer Loyalty & Service Quality",
        110: "Digital Transformation & Dynamic Capabilities",
        111: "High-Speed Railway & Regional Development using Difference in Difference Methods",
        112: "Epidemiological Causal Inference & Public Health",
        113: "Bayesian Network Learning & Graphical Models",
        114: "Mendelian Randomization in Neurological & Psychiatric Disorders",
        115: "Leadership Styles, Ethical Leadership & Employee Creativity",
        116: "PTSD ",
        117: "Islamic Banking & FinTech Adoption in Indonesia",
        118: "Retirement, Pension Reform & Health Effects using Regression Discontinuity Designs",
        119: "Spinal Surgery and Outcomes using Propensity Score Methods",
        120: "Renewable Energy Adoption & Energy Saving Behavior",
        121: "Patient Satisfaction & Healthcare Service Quality",
        122: "Renal Cell Carcinoma Nephrectomy & Outcomes using Propensity Score Methods",
        123: "Electric Vehicle Adoption Intention",
        124: “-1”,
        125: “-1”,
        126: "Corporate Governance, CEO Compensation & Firm Performance",
        127: "Career Adaptability, Employability & Career Development",
        128: "Hepatitis, Antiviral Therapy & Risk",
        129: "Implicit Causality in Linguistics: Pronouns & Discourse",
        130: "Information and Communication Technology, Internet Access & Economic Growth",
        131: "Hepatocellular Carcinoma Surgery & Outcomes using Propensity Score Methods",
        132: "Immigration, Migration & Labour Market Integration",
        133: "Thyroid Cancer Surgery and Outcomes using Propensity Score Methods",
        134: "Regression Discontinuity Design: Methods & Inference",
        135: "Urban Green Space, Parks & Mental Health",
        136: "Corruption, Anti-Corruption Policies & Economic Growth",
        137: "Blockchain Technology Adoption",
        138: "Vaccine Hesitancy & Vaccination Intention",
        139: "Ophthalmic Surgery: Cataract, Glaucoma & Retinal Disorders",
        140: "Working Memory, Executive Function & Cognitive Ability",
        141: "Corporate Social Responsibility & Customer Loyalty",
        142: "Tax Compliance & Tax Evasion",
        143: "Statin Therapy & Cardiovascular Outcomes",
        144: "IoT, Mobile Commerce & Smart Home Adoption",
        145: "Total Quality Management & Organizational Performance",
        146: "Online Shopping, Trust & Repurchase Intention",
        147: "Mindfulness, Gratitude & Psychological Well-being",
        148: "Coronary Stenting Methods and Outcomes",
        149: "Health Expenditure, Economic Growth & Population Health",
        150: "Childbirth Delivery Types & Outcomes using Propensity Score Methods",
        151: "Agricultural Commodity Prices, Cointegration & Granger Causality",
        152: "Student Satisfaction & University Service Quality",
        153: "Financial Literacy, Investment Decisions & Behavioral Biases",
        154: "Drug-Induced Liver Injury & Causality Assessment",
        155: "Oral Anticoagulants: Warfarin vs. DOACs in Atrial Fibrillation",
        156: "Causal Mediation Analysis & Moderated Mediation",
        157: "Inflammatory Bowel Disease: Crohn's, Colitis & Biologic Therapies",
        158: "Work Engagement, Organizational Commitment & Perceived Support",
        159: "Antipsychotic Medications & Schizophrenia Treatment",
        160: "Gallbladder & Biliary Surgery",
        161: "International Remittances, Migration & Poverty Reduction",
        162: "IVF, Embryo Transfer & Reproductive Outcomes",
        163: "Fuzzy Cognitive Maps",
        164: "Board Gender Diversity & Firm Performance",
        165: "Atrial Fibrillation: Catheter Ablation",
        166: "Emotional Intelligence, Leadership & Employee Performance",
        167: "Maternal & Neonatal Health: Antenatal Care",
        168: "Human Resource Management & Organizational Performance",
        169: "Online Gaming, Gamification & Purchase Intention",
        170: "Head & Neck Cancer: Radiotherapy & Chemotherapy",
        171: "Causal Reasoning in Child Development",
        172: "Work-Family Conflict & Life Balance",
        173: "Accounting Information Systems & Financial Reporting",
        174: "Cannabis Legalization, Medical Use & Psychosis Risk",
        175: "Augmented/Virtual Reality in Tourism",
        176: "Childhood Asthma, Allergies & Atopic Dermatitis",
        177: "Cybersecurity Behavior & Information Security Compliance",
        178: "Bladder Cancer: Radical Cystectomy",
        179: "Social & Political Trust, Civic Engagement",
        180: "IoT Routing Protocols & Sensor Networks",
        181: "Cryptocurrency Markets & Bitcoin Price Volatility",
        182: "Workplace Spirituality & Spiritual Leadership",
        183: "Organizational Agility & Dynamic Capabilities",
        184: "Personality Traits & Psychometric Assessment",
        185: "Donation Behavior & Charitable Giving",
        186: "Grundtvig, Folk High Schools & Danish Theology",
        187: "Diabetes Self-Management & Glycemic Control",
        188: "Colorectal Cancer Chemotherapy & Outcomes using Propensity Score Methods",
        189: "Dividend Policy, Capital Structure & Firm Value",
        190: "Colorectal Surgery & Outcomes using Propensity Score Methods",
        191: "Multiple Sclerosis: Disease-Modifying Therapies",
        192: "Microfinance, Women Empowerment & Poverty Reduction",
        193: "Migraine ",
        194: "Opioid Prescribing & Opioid Use Disorder",
        195: "Postpartum Depression & Maternal Mental Health",
        196: "E-Government Adoption & Digital Public Services",
        197: "AI Chatbots & Customer Service",
        198: "Student Creativity & Creative Self-Efficacy",
        199: "Elderly Mental Health, Social Participation & Life Satisfaction",
        200: "Environmental, Social, and Governance Disclosure, Sustainability & Firm Value",
        201: "Loneliness, Social Isolation & Depression in Older Adults",
        202: "AI Adoption in Human Resource Management & Recruitment",
        203: "Peacekeeping Operations, Conflict & Political Science",
        204: "Cloud Computing Adoption & SaaS",
        205: "ADHD: Diagnosis & Interventions in Children & Adults",
        206: "Big Data Analytics & Supply Chain Strategy",
        207: "COPD & Respiratory Outcomes",
        208: "Rheumatoid Arthritis: TNF Inhibitors & Methotrexate",
        209: "Luxury Brand Consumption ",
        210: "Online Food Delivery & Customer Satisfaction",
        211: "Lean Manufacturing & Operational Performance",
        212: "Causal Inference in Recommender Systems",
        213: "Renin-Angiotensin System & Antihypertensive Therapy",
        214: "Military Expenditure, Defense Spending & Economic Growth",
        215: "Cardiac Arrest, CPR & Resuscitation Outcomes",
        216: "DAGs in Clinical & Epidemiological Contexts",
        217: "School Bullying & Victimization",
        218: "Advertising Effectiveness & Ad Avoidance",
        219: "Multisensory Perception & Bayesian Causal Inference",
        220: "Marital Satisfaction & Romantic Relationships",
        221: "Cyber Attacks, Intrusion Detection & Malware",
        222: "Proton Pump Inhibitors & Gastric Cancer Risk using Propensity Score Methods",
        223: "Cervical & Ovarian Cancer Surgery Survival using Propensity Score Methods",
        224: "Brain Tumor Surgery & Survival using Propensity Score Methods",
        225: "Organizational Citizenship Behavior & Servant Leadership",
        226: "Enterprise Resource Planning Impacts",
        227: "Rectal Cancer Surgery & Outcomes using Propensity Score Methods",
        228: "Terrorism & Fear of Terror",
        229: "Music Education & Self-Efficacy",
        230: "Stigma & Help-Seeking Behavior",
        231: "Vaccine Safety & Adverse Events Following Immunization",
        232: "-1",
        233: "Materials Science: Alloys, Coatings & Diffusion",
        234: "Online Privacy & Self-Disclosure",
        235: "Academic Procrastination & Self-Regulation",
        236: "Problem Gambling, Gaming Addiction & Loot Boxes",
        237: "Hernia Repair Surgery",
        238: "Solar-Terrestrial Physics & Geomagnetic Storms",
        239: "Volunteering, Volunteer Motivation & Social Capital",
        240: "Family Business Governance & Succession Planning",
        241: "Intellectual Property & Patent Policy",
        242: "Sleep Disorders & Mendelian Randomization",
        243: "Diabetes Disease Management Programs using Difference in Difference Methods",
        244: "Helicobacter Pylori & Gastric Ulcer Treatment",
        245: "Quantile Regression & Instrumental Variable Quantile Methods",
        246: "Perfectionism: Adaptive vs. Maladaptive",
        247: "Intrahepatic Cholangiocarcinoma Surgery & Prognosis using Propensity Score Methods",
        248: "Cyberbullying & Adolescent Cyber Victimization",
        249: "COVID-19 Pandemic Policies & Epidemic Control",
        250: "Juvenile Delinquency & Peer Influence",
        251: "Counterfactual Reasoning & Causal Inference",
        252: "Endoscopic Submucosal Dissection (ESD)",
        253: "AI-Driven Personalization & Digital Marketing",
        254: "Religiosity, Secularization & Sociology of Religion",
        255: "Breastfeeding Practices",
        256: "Child Malnutrition, Stunting & Feeding Practices",
        257: "Employee Turnover Intention & Job Satisfaction",
        258: "Service Robots & Robot Acceptance in Hospitality",
        259: "Causal Inference Methods & Big Data",
        260: "Hearing Loss, Tinnitus & Vestibular Disorders",
        261: "Dynamic Panel Data Models: GMM & Instrumental Variables",
        262: "Malaria Prevention & Chemoprevention",
        263: "Food Tourism & Culinary Tourism",
        264: "Legal Causation, Medical Malpractice & Liability",
        265: "Social Anxiety, Anxiety Disorders & Emotion Regulation",
        266: "Fake News, Journalism & Social Media",
        267: "Support Vector Machines, Multiclass Classification & DAGs",
        268: "Organizational Citizenship Behavior & Organizational Justice",
        269: "Vaccine-Associated Adverse Neurological Events",
        270: "Digital Library Services & User Satisfaction",
        271: "Education Expenditure, Economic Growth & Granger Causality",
        272: "Algorithmic, Counterfactual Fairness & Discrimination",
        273: "Schizophrenia, Neurocognition & Social Functioning",
        274: "Innovative Work Behavior & Transformational Leadership",
        275: "Environmental Kuznets Curve & Ecological Footprint",
        276: "Alzheimer's Disease: Neurodegeneration & Amyloid Pathology",
        277: "Intracranial Aneurysm Treatments & Outcomes using Propensity Score Methods",
        278: "Colorectal Liver Metastasis Surgery Outcomes using Propensity Score Methods",
        279: "Aortic Aneurysm & Endovascular Repair",
        280: "Hume's Philosophy of Causality",
        281: "Sustainable Fashion & Apparel Consumption",
        282: "Bond Graph Modeling & Causality in Physical Systems",
        283: "Digital Leadership & Employee Digital Competence",
        284: "Phylogenetic Networks & Evolutionary Genomics",
        285: "Causality & Econometrics",
        286: "-1",
        287: "Vocational Rehabilitation, Compulsory Schooling & Employment",
        288: "Pulmonary Embolism & Venous Thromboembolism",
        289: "Periodontal Disease, Tooth Loss & Obesity",
        290: "Osteoporosis Treatment: Bisphosphonates & Fracture Prevention",
        291: "Tuberculosis Treatment & MDR-TB",
        292: "Wine Tourism & Wine Consumer Behavior",
        293: "Dementia Risk & Alzheimer's Prevention",
        294: "Bariatric Surgery & Weight Loss",
        295: "Causal Representation Learning & Domain Generalization",
        296: "Neural Architecture Search & Deep Neural Networks",
        297: "Kantian Philosophy of Causality",
        298: "Urological Stone Disease: Nephrolithotomy & Ureteroscopy",
        299: "Internet Use, Social Participation & Depression in Elderly",
        300: "Difference-in-Differences: Parallel Trends & Staggered Adoption"
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
