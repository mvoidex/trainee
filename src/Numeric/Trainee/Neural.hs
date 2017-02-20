{-# LANGUAGE GADTs, FlexibleContexts, RankNTypes #-}

module Numeric.Trainee.Neural (
	Net, net, netSeed,
	(⭃),
	ndup, npar, ndupWith, nsum, nfold,
	fc, conv, conv2,
	dconv, dconv2,

	sigma, relu, softplus,
	summator, biaser, activator, convolver, convolver2,
	normVar,

	module Numeric.Trainee.Types,
	module Numeric.Trainee.Learnee
	) where

import Prelude hiding ((.), id)
import Prelude.Unicode

import Control.Monad (replicateM, liftM2)
import Data.Random
import qualified Data.Vector as V
import Numeric.LinearAlgebra hiding (conv, conv2)
import System.Random (newStdGen, StdGen)

import Numeric.Trainee.Types
import Numeric.Trainee.Gradee
import Numeric.Trainee.Learnee

type Net a = Learnee (Vector a) (Vector a)

net ∷ RVar a → IO a
net n = netSeed <$> newStdGen <*> pure n

netSeed ∷ StdGen → RVar a → a
netSeed g n = fst $ sampleState n g

(⭃) ∷ RVar (Learnee a b) → RVar (Learnee b c) → RVar (Learnee a c)
n ⭃ l = liftM2 (⇉) n l

-- | Split layers, just wraps @vdup@
ndup ∷ Num a ⇒ Int → RVar (Learnee a (V.Vector a))
ndup = return ∘ computee ∘ vdup

-- | Process vector of layers
npar ∷ Int → RVar (Learnee a b) → RVar (Learnee (V.Vector a) (V.Vector b))
npar n = fmap (parallel ∘ V.fromList) ∘ replicateM n

-- | Split layers, justs wraps @vdupWith@
ndupWith ∷ (a → a → a) → Int → RVar (Learnee a (V.Vector a))
ndupWith fn = return ∘ computee ∘ vdupWith fn

-- | Join layers with sum
nsum ∷ Num a ⇒ RVar (Learnee (V.Vector a) a)
nsum = return (computee vsum)

-- | Join layers with custom sum
nfold ∷ (a → a → a) → RVar (Learnee (V.Vector a) a)
nfold = return ∘ computee ∘ vfold

-- | Fully connected layer
fc ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → Int → RVar (Net a)
fc f inputs outputs = do
	s ← summator inputs outputs
	b ← biaser outputs
	return $ s ⇉ b ⇉ activator f

-- | Convolution layer 1-d
conv ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → RVar (Net a)
conv f w = do
	c ← convolver w
	b ← do
		bs ← normVar
		return $ learnee biasVec bs
	return $ c ⇉ b ⇉ activator f

-- | Convolution layer 2-d
conv2 ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Unary (Matrix a) → (Int, Int) → RVar (Learnee (Matrix a) (Matrix a))
conv2 f (w, h) = do
	c ← convolver2 w h
	b ← do
		bs ← normVar
		return $ learnee biasMat bs
	return $ c ⇉ b ⇉ activator f

-- | Depth (with several input and output channels) convolution 1-d
dconv ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Unary (Vector a) → Int → Int → Int → RVar (Learnee (V.Vector (Vector a)) (V.Vector (Vector a)))
dconv f inputs outputs width =
	ndupWith (V.zipWith (+)) outputs ⭃
	npar outputs (npar inputs (conv f width) ⭃ nsum)

-- | Depth (with several input and output channels) convolution 2-d
dconv2 ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Unary (Matrix a) → Int → Int → (Int, Int) → RVar (Learnee (V.Vector (Matrix a)) (V.Vector (Matrix a)))
dconv2 f inputs outputs (w, h) =
	ndupWith (V.zipWith (+)) outputs ⭃
	npar outputs (npar inputs (conv2 f (w, h)) ⭃ nsum)

sigma ∷ Floating a ⇒ a → a
sigma t = 1 / (1 + exp (negate t))

relu ∷ Fractional a ⇒ a → a
relu t = 0.5 * (1 + signum t) * t

softplus ∷ Floating a ⇒ a → a
softplus t = log (1 + exp t)

summator ∷ (Distribution Normal a, Fractional a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Int → Int → RVar (Net a)
summator inputs outputs = do
	ws ← replicateM (inputs * outputs) normVar
	return $ learnee matVec ((outputs >< inputs) ws)

biaser ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Int → RVar (Net a)
biaser sz = do
	bs ← replicateM sz normVar
	return $ learnee odot (fromList bs)

activator ∷ Num a ⇒ Unary a → Learnee a a
activator f = computee (unary f)

convolver ∷ (Distribution Normal a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Int → RVar (Net a)
convolver w = do
	ws ← replicateM w normVar
	return $ learnee corrVec (fromList ws)

convolver2 ∷ (Distribution Normal a, Fractional a, Numeric a, Parametric (Vector a), Parametric a) ⇒ Int → Int → RVar (Learnee (Matrix a) (Matrix a))
convolver2 w h = do
	ws ← replicateM (w * h) normVar
	return $ learnee corrMat ((w >< h) ws)

normVar ∷ (Distribution Normal a, Fractional a) ⇒ RVar a
normVar = normal 0.0 0.25