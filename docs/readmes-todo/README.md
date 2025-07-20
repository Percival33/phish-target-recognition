# User-Focused Documentation

This directory contains improved, user-focused versions of the project's README files. These were created in response to feedback that the original documentation was too technical and difficult for end users to follow.

## Available Guides

### 📖 [Main User Guide](README-user-guide.md)
**Recommended starting point for all users**

A complete rewrite of the main README with:
- **Clear use-case paths** (A, B, C, D) based on what users want to accomplish
- **Step-by-step workflows** with detailed explanations
- **Integrated cross-validation** as part of the main evaluation process
- **Comprehensive troubleshooting** section
- **User-focused language** instead of developer-focused technical details

### 🔧 [Baseline Model Guide](baseline-user-guide.md)
**For users working with the baseline phishing detection model**

Simplified guide covering:
- **Quick testing** scenarios for immediate results
- **Cross-validation integration** with the main workflow
- **Configuration and tuning** guidance
- **Performance expectations** and limitations
- **Clear use-case scenarios** (CV evaluation, quick testing, custom datasets)

### 📊 [Evaluation Guide](eval-user-guide.md)
**For users running statistical evaluation and analysis**

Comprehensive evaluation guide with:
- **Three clear paths** for different evaluation scenarios
- **Complete workflow** from data preparation to statistical analysis
- **Configuration explanations** (not just listings)
- **Results interpretation** guidance
- **Custom dataset preparation** step-by-step

## Key Improvements Made

### ✅ Addressed Original Feedback Issues

1. **"próbuję jakoś rozczytać co mam zrobić"** (hard to understand what to do)
   - → Added clear use-case paths with step-by-step instructions

2. **"nie rozumiem jak przygotować zbiory danych"** (don't understand dataset preparation)
   - → Dedicated sections for data preparation with complete examples

3. **"nie wiem do czego się odnosi"** (don't know what refers to what)
   - → Added context and explanations for each script and command

4. **"konfiguracja CV"** (CV configuration unclear)
   - → CV properly integrated into main workflow, not hanging separately

5. **"dokumentacja skierowana do dewelopera"** (documentation aimed at developers)
   - → Rewritten for end users with practical focus

### 🎯 User Experience Improvements

- **Immediate orientation** with "What do you want to do?" sections
- **Progressive complexity** from simple to advanced scenarios
- **Complete workflows** that users can follow from start to finish
- **Troubleshooting guidance** for common issues
- **Clear expectations** about what each step produces

## How to Use These Guides

### For First-Time Users
Start with the [Main User Guide](README-user-guide.md) and follow **Path A: Complete Model Evaluation**.

### For Specific Components
- **Baseline model only**: See [Baseline Model Guide](baseline-user-guide.md)
- **Evaluation and statistics**: See [Evaluation Guide](eval-user-guide.md)
- **Custom datasets**: All guides have dedicated sections for this

### For Advanced Users
- **Path C** in the main guide for reproducing paper results
- **Custom Dataset** sections in all guides for your own data

## Relationship to Original Documentation

These guides **supplement** the original technical documentation. They provide:
- **User-focused workflows** (these guides)
- **Technical reference** (original READMEs)
- **Implementation details** (code comments and docstrings)

The original documentation remains valuable for developers and advanced users who need technical details.

## Feedback Implementation

These guides directly address the feedback about:
- Unclear step-by-step processes
- Confusing external dataset preparation
- Missing cross-validation integration
- Unclear model execution flow
- Too technical/developer-focused language

The result is documentation that guides users through complete workflows rather than just listing available commands.
