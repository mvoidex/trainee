{-# LANGUAGE RankNTypes, FlexibleContexts, FlexibleInstances, OverloadedStrings #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module Numeric.Trainee.Learnee (
	eval,
	(⇉), (‖), into, paired, parallel,
	loss, regularization, squared, logLoss, crossEntropy, lossL1, lossL2,
	learnee, computee,

	makeBatches, shuffleList,

	miss, avg,
	runLearnT, runLearn,
	learnPass, trainOnce, trainBatch, trainEpoch, trainUntil
	) where

import Prelude.Unicode

import Control.Arrow ((***), (>>>), first)
import Control.DeepSeq
import Control.Lens
import Control.Monad.State.Strict
import Control.Monad.Writer.Strict
import Data.List (unfoldr)
import Data.Random (runRVar, StdRandom(..), MonadRandom)
import Data.Random.Internal.Source
import Data.Random.List (shuffle)
import qualified Data.Vector as V
import Numeric.AD (AD)
import Numeric.AD.Mode.Forward (Forward, auto, diff')

import Numeric.Trainee.Types hiding (Ow(..))

eval ∷ Learnee a b → a → b
eval (Learnee ws f) = fst ∘ f ws

(⇉) ∷ Learnee a b → Learnee b c → Learnee a c
(⇉) = (>>>)

into ∷ Learnee a b → Learnee b c → Learnee a c
into = (>>>)

(‖) ∷ Learnee a b → Learnee a' b' → Learnee (a, a') (b, b')
Learnee lws f ‖ Learnee rws g = lws `deepseq` rws `deepseq` Learnee (Params (lws, rws)) (h ∘ castParams) where
	h (lws', rws') (x, y) = x `seq` y `seq` x' `seq` y' `seq` lws' `deepseq` rws' `deepseq` ((x', y'), up) where
		(x', f') = f lws' x
		(y', g') = g rws' y
		up (dx', dy') = dx' `seq` dy' `seq` dx `seq` dy `seq` lws'' `deepseq` rws'' `deepseq` (Params (lws'', rws''), (dx, dy)) where
			(lws'', dx) = f' dx'
			(rws'', dy) = g' dy'

paired ∷ Learnee a b → Learnee a' b' → Learnee (a, a') (b, b')
paired = (‖)

parallel ∷ V.Vector (Learnee a b) → Learnee (V.Vector a) (V.Vector b)
parallel ls = ls `deepseq` Learnee (Params wss) (h ∘ castParams) where
	wss = V.map (view params) ls
	fs = V.map (view forwardPass) ls
	h wss' xs = xs `seq` ys `seq` wss' `deepseq` (ys, up) where
		(ys, gs) = V.unzip $ V.zipWith3 id fs wss' xs
		up ds = first Params $ V.unzip $ V.zipWith id gs ds

loss ∷ Num a ⇒ (forall s . AD s (Forward a) → AD s (Forward a) → AD s (Forward a)) → Loss a
loss fn y' = diff' (fn (auto y'))

regularization ∷ Num a ⇒ (forall s . AD s (Forward a) → AD s (Forward a)) → Regularization a
regularization fn = diff' fn

squared ∷ Floating a ⇒ Loss a
squared = loss $ \y' y → (y - y') ** 2

logLoss ∷ Floating a ⇒ Loss a
logLoss = loss $ \y' y → - (y' * log y)

crossEntropy ∷ Floating a ⇒ Loss a
crossEntropy = loss $ \y' y → - (y' * log y + (1 - y') * log (1 - y))

lossL1 ∷ Floating a ⇒ Regularization a
lossL1 = regularization abs

lossL2 ∷ Floating a ⇒ Regularization a
lossL2 = regularization (**2)

learnee ∷ Parametric w ⇒ Gradee (w, a) b → w → Learnee a b
learnee g ws = Learnee (Params ws) (h ∘ castParams) where
	h ws' x = x `seq` ws' `deepseq` y `seq` (y, back) where
		y = view (runGradee g) (ws', x)
		back dy = dy `seq` dx `seq` dws `deepseq` (Params dws, dx) where
			(dws, dx) = set (runGradee g) dy (ws', x)

computee ∷ Gradee a b → Learnee a b
computee g = Learnee (Params NoParams) h where
	h _ x = x `seq` y `seq` (y, back) where
		y = view (runGradee g) x
		back dy = dy `seq` dx `seq` (Params NoParams, dx) where
			dx = set (runGradee g) dy x

makeBatches ∷ Int → [a] → [[a]]
makeBatches sz = takeWhile (not ∘ null) ∘ unfoldr (Just ∘ splitAt sz)

shuffleList ∷ MonadRandom m ⇒ [a] → m [a]
shuffleList ls = runRVar (shuffle ls) StdRandom

miss ∷ HasNorm b ⇒ Learnee a b → Loss b → Sample a b → Norm b
miss l c s = snd $ learnPass l c s

avg ∷ (Foldable t, Fractional a) ⇒ t a → a
avg ls = sum ls / fromIntegral (length ls)

runLearnT ∷ Learnee a b → StateT (Learnee a b) m c → m (c, Learnee a b)
runLearnT = flip runStateT

runLearn ∷ Learnee a b → State (Learnee a b) c → (c, Learnee a b)
runLearn = flip runState

learnPass ∷ HasNorm b ⇒ Learnee a b → Loss b → Sample a b → (Params, Norm b)
learnPass (Learnee ws f) c (Sample x y') = x `seq` ws `deepseq` y' `seq` y `seq` dws `deepseq` (dws, norm e) where
	(y, back) = f ws x
	(e, de) = c y' y
	(dws, _) = back de

trainOnce ∷ (MonadState (Learnee a b) m, HasNorm b) ⇒ Rational → Loss b → Sample a b → m (Norm b)
trainOnce λ c s = state train' where
	train' l = dw `deepseq` (e, over params (subtract (fromRational λ * dw)) l) where
		(dw, e) = learnPass l c s

trainBatch ∷ (MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b)) ⇒ Rational → Loss b → Samples a b → m (Norm b)
trainBatch λ c xs = state train' where
	train' l = dw `deepseq` (e, over params (subtract (fromRational λ * dw)) l) where
		(dw, e) = (avg *** avg) ∘ V.unzip ∘ V.map (learnPass l c) $ xs

trainEpoch ∷ (MonadRandom m, MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b)) ⇒ Rational → Int → Loss b → Samples a b → m (Norm b)
trainEpoch λ batch c xs = do
	ix' ← shuffleList [0 .. V.length xs - 1]
	fmap (avg ∘ concat) ∘ mapM (\ixs → fmap (replicate (length ixs)) (trainBatch λ c (samples' ixs xs))) $ makeBatches batch ix'
	where
		samples' is vs = samples $ map (vs V.!) is

instance (Monoid w, MonadRandom m) ⇒ MonadRandom (WriterT w m) where
	getRandomPrim = lift ∘ getRandomPrim

instance MonadRandom m ⇒ MonadRandom (StateT s m) where
	getRandomPrim = lift ∘ getRandomPrim

trainUntil ∷ (MonadRandom m, MonadState (Learnee a b) m, HasNorm b, Fractional (Norm b), Ord (Norm b)) ⇒ Rational → Int → Int → Norm b → Loss b → Samples a b → m (Norm b)
trainUntil λ epochs batch eps c xs = do
	bs ← execWriterT $ train' epochs
	return $ last bs
	where
		train' 0 = return ()
		train' n = do
			e ← trainEpoch λ batch c xs
			tell [e]
			unless (e < eps) $ train' (pred n)
