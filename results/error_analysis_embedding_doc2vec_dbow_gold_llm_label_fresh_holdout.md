# Error Analysis — embedding_doc2vec_dbow_gold_llm_label_fresh_holdout

166 false positives and 109 false negatives out of 772 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `neutral_allergen_mention` | 95 | 57% | 137 |
| `illness_mentioned_not_caused_here` | 32 | 19% | 136 |
| `unexplained_fp` | 18 | 11% | 187 |
| `generic_complaint_no_hazard` | 11 | 7% | 98 |
| `negated_hazard` | 7 | 4% | 143 |
| `secondhand_or_hearsay` | 2 | 1% | 264 |
| `strong_negative_sentiment_only` | 1 | 1% | 60 |

### `neutral_allergen_mention` — 95 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- I love the food here, it makes for a different gluten free meal in unique surroundings. My only issue is the inconsistent opening hours, they frequently close early, and some weeks don't open at all. The owner/chef is very rude, has zero customer service ability, or maybe just di… (p=0.321)
- It doesn't look like much, but this local place was great for take out. Due to COVID-19, we haven't been going out much, but got real tired of cooking after a whole month. We wanted to support a local business, and were able to have a no contact pick up. Win win all around. I ord… (p=0.697)
- Hoe excited was i to find a gluten free vegan pizza (up charge) option. Staff was super accommodating and courteous. When ordering gluten free it is common to be asked Allergy or Preferance. But when i get down the line to add my toppings they actually switched gloves to prevent … (p=0.535)

### `illness_mentioned_not_caused_here` — 32 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- Ignorant and inconsiderate!!! Do not even attempt to eat here if you have any type of food allergy or dietary restriction. They are completely inconsiderate of making any accommodations. my friend took her daughter here who is lactose intolerant. She brought her own almond milk f… (p=0.624)
- Ice cream was awsome but more importantly my daughter has severe food allergies. Staff were knowledgeable and had a good process preventing cross contamination.… (p=0.647)
- Wowza. Someone doesn't often expect any South Philly establishment off the beaten path (aka Passyunk) to serve non-food-poisoning-containing raw fish (SPTR aside). But Hibachi2Go breaks the mold and serves decently fresh sashimi, delicious rolls and a slew of tasty salads/teriyak… (p=0.499)

### `unexplained_fp` — 18 cases

No rule matched — requires manual review.

- Peruvian Apology! I'm peruvian and this place has nothing to do with Peruvian food, sorry Gringos but this place is an scam is more Texmex than Peruvian. I ordered lomo saltado because no peruvian will mess up that dish but surprise they f..up royally, meat burned and way too muc… (p=0.876)
- I've been to the Loews on multiple occasions. I've had a wonderful time each time i've gone there. The most recent time I went there I brought my two springer spaniel pups. I was so excited that Loews allows pets. Most of the other pet friendly hotels in philly are not nice. They… (p=0.681)
- They serve burnt black fajitas. Some staff have rude and have bad attitude. I did not liked the service. When complaint about the burnt fajitas the staffs says it is instructions received by their corporate. Cannot believe who will like burnt fajitas. It has no nutritional value … (p=0.789)

### `generic_complaint_no_hazard` — 11 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- When you're on a time crunch, the last thing you need is a slow moving line in a fast food drive-through. As we inched through this particular Steak 'n Shake, we all immediately regretted our decision on choosing this place. Due to being squeezed between two other vehicles, we we… (p=0.551)
- Nice clean store with great selection of organics. People at the register are always friendly too and love that their bakery super careful about cross contamination when it comes to peanuts...at least so they say when I expressed the concern.… (p=0.489)
- When it comes to fresh sushi and authentic Asian cuisine dining in the suburbs. I definitely recommend this place. Staff are very friendly and accommodating to our requests. Their special rolls are delicious! With one bite of the lobster roll, my mouth was bursting with exciting … (p=0.568)

### `negated_hazard` — 7 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Food is over priced for what you get and server gave me the wrong sandwich! Good thing I'm not allergic to the turkey I didn't want The bread was toasted so much my nePhew cut his mouth! Wait in line and for food is too long! Not worth it at all!!!!!!… (p=0.351)
- After reading reviews that Green Basil happily accommodates vegan and vegetarian diets, I ordered delivery specifying that everything must be vegan like this: (***** VEGAN PLEASE *****). Normally if a restaurant cannot accommodate this, they have the courtesy of contacting me to … (p=0.389)
- VEGANS BEWARE! The food I ate was fresh. The place is friendly and clean. But vegans beware. I was told the flan from oopsy Daisy's was vegan and it's not. I found out from ms daisy after eating it unfortunately. Good thing I'm not allergic. Wish they'd be a little more mindful i… (p=0.999)

### `secondhand_or_hearsay` — 2 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- A full celebration tea for 2 contains: 2 pots of tea, a fruit plate,salad and soup, 2 tier tea stand and cupcake with icecream as dessert. The environment is very well decorated , and apparently it's someone's home. When I made a reservation they told me there's only 12 and 1 o'c… (p=0.427)
- The restaurant was inconsistent at best. I went for the lunch buffet. The food was actually pretty good. The food was tasty and fish was fresh. The sashimi was actually normal thickness. There were also a hot food selection. The problem was there isn't that many people that go fo… (p=0.987)

### `strong_negative_sentiment_only` — 1 cases

Highly negative review with no hazard content. The model is partly reading sentiment as hazard, unsurprising given the label used a star-rating gate.

- If I could give this 0 stars it would be more appropriate! Having Celiac is always a struggle to eat out, but I've always managed to find something to eat at every restaurant. Today I was told I could not eat anything in this restaurant!! Really??? Are you just lazy??? I should b… (p=0.958)

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 60 | 55% | 192 |
| `buried_in_long_review` | 17 | 16% | 266 |
| `unsafe_handling_no_illness` | 12 | 11% | 230 |
| `positive_review_with_hazard` | 11 | 10% | 165 |
| `negation_misread` | 6 | 6% | 363 |
| `contamination_no_illness` | 2 | 2% | 166 |
| `mild_understated_wording` | 1 | 1% | 459 |

### `unexplained_fn` — 60 cases

No rule matched — requires manual review.

- So before I came to this place I came on yelp to review it, and while the reviews were basically 50/50 I decided to give them a chance. First, it took us 45 minutes to finally get our appetizers. A guest in our party didn't get the correct order. She specifically asked for no tom… (p=0.005)
- On 2/20/16, I encountered this place that my friends raved about. Once we were seated the waitress brought water to the table OMG the PLASTIC cup was cloudy, Grease was floating on top of my water made me want to VOMIT. I asked for a to go cup of water she said those cups are not… (p=0.178)
- Chase appears nice on the outside, but the food was mediocre at best, certainly not worth the price. My mom got terrible food poisoning from the ravioli that kept her up all night. The vodka pappardelle and eggplant parm were good, but swimming in sauce. If you are considering It… (p=0.091)

### `buried_in_long_review` — 17 cases

Hazard mentioned late in a long review, or diluted by surrounding content. NOTE: the tempting 'past the 256-token window' reading was MEASURED AND REFUTED for the gold false negatives — with the real DebertaV2TokenizerFast at max_length=256, only 1 of 23 residual FNs has its hazard cue past the window, only 2 of 23 exceed 256 tokens at all, and the median cue position is token 39. Raising max_length would recover at most one FN. See results/gold_fn_handread.md. Treat this bucket as dilution/salience, not truncation.

- I ordered a pan pizza. I trusted them. They are professionals. They do this all the time. Well, this isn't what I experienced. I ordered half green pepper, tomato and olive. What did I get? No olive. Onion substitute instead. I bit into it, in a darkened room, and immediately che… (p=0.021)
- My Husband and I visit PF Chang yesterday for dinner. My husband ordered the Kung Pao chicken with ice tea and I ordered the pepper steak, for appetizer we ordered the crispy green beans. The crispy green beans was amazing and the drinks (coconut cooler) was really good. My husba… (p=0.002)
- I am from Lakeland and visited the Tavern yesterday with 8 of our friends. The stuff on the walls are cool, and drinks weren't bad tasting. However - it took about 20 minutes to be acknowledged after we were seated. We were given our waters, and ordered a first round of drinks, w… (p=0.146)

### `unsafe_handling_no_illness` — 12 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- This review is strictly about the food: it pains me to do this but it needs to be known. I'll start with the positives; The french fries were perfect as well as the cheese and charcuterie plate. The Mac and cheese was bland even with sausage and tasted of only cream (I tried some… (p=0.198)
- I have been to this restaurant twice now. The first time was about as perfect as you can imagine. The food was perfect the prime rib perfectly cooked and seasoned, the waitstaff helpful and polite. Literally everything was perfect. It was probably the best prime rib I have ever h… (p=0.000)
- They were selling expired baby yogurt. I bought a pack of Stonyfield organic yogurt for my 11 month old twins on Dec. 30th. I gave my son some and he didn't seems to care for it, and my daughter gagged on it and threw up. Turns out it expired Dec. 13th! I returned to the store on… (p=0.074)

### `positive_review_with_hazard` — 11 cases

4-5 star review reporting a hazard. The star gate in the labelling rule means the training data barely contains these, so the model associates hazards with low ratings.

- ***This is more of a 3.5 place but I rounded up*** Okay, so I was in Nashville last weekend for a friend's bachelorette party. For our first night in town, we came here for dinner. I'll start by saying that the building itself is super swanky. It looked small from the outside but… (p=0.020)
- I took my girlfriend here and she got food poisoning lmao but the food was bomb tho ‍… (p=0.117)
- Really impressed at El Sur today. Inconspicuous, colorful building - definitely one of those Don't judge a book by its cover scenarios. Very friendly staff and always smiling. I ordered the fish tacos and my husband ordered the shredded beef tacos. The waitress warned me about th… (p=0.069)

### `negation_misread` — 6 cases

Negation cue present but the review is genuinely a hazard ('not the first time I got sick here'). Over-application of the negation pattern.

- I'm not sure what I expected before walking in which left me open to what was in store. I expect 25% experience/environment, 25% service, 25% Food Quality and 25% cost/value. It's located downtown, and has a full bar in the front, with room to have appetizers and loiter until get… (p=0.001)
- You know, it's really too bad that I have to give the Hookah House a three-start review. The place is beautiful, and I've often said that it seems like a place that James Bond would have visited. If Bond were to have visited the Hookah House, he'd have enjoyed a seat in one of th… (p=0.000)
- This establishment was by far one of my worst experiences as a customer. After, entering the establishment and giving the hostess our name to be seated we waited about 2-3 minutes (great!). However, the individual who was going to seat us got our name incorrect and called it 2-3 … (p=0.000)

### `contamination_no_illness` — 2 cases

A foreign object or tampering, without illness vocabulary. Same inherited blind spot as above; this is where the heuristic's recall is weakest (88.5% on contamination).

- The first time we wen to Izzos it was ok, pricey for a fast food place. The second time we went was awful. The girl who made our burritos was rude telling my husband to "relax" when he said he didn't need a cup for his bottled drink. She thought she was funny, she wasn't. She als… (p=0.032)
- I go here all the time, it's one of my late night faves. But tonight I went there, ordered my usual with one of my friends. Pulled up to the window and asked for 2 water cups. We got our water cups and took a sip. They were basically carbonated sink water or sprite with zero flav… (p=0.160)

### `mild_understated_wording` — 1 cases

Understated symptoms ('didn't sit right with me'). Same ceiling as above, milder.

- I've been wanting to check this place out for for brunch for months now. I finally did and man am I disappointed. The dining experience got worse and worse with every passing moment. First-the $12 Bloody Mary "bar." For $12 I thought there would be more to the make your own blood… (p=0.000)
